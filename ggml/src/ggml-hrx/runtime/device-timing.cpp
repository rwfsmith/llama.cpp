#include "runtime/device-timing.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
#include "hrx_internal.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#endif

namespace ggml::hrx {
namespace {

static bool environment_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool using_hip_driver() {
    const char * driver = std::getenv("HRX_GPU_DRIVER");
    return driver != nullptr && std::strcmp(driver, "hip") == 0;
}

struct DeviceTimingCounters {
    std::atomic<uint64_t> init_attempts{ 0 };
    std::atomic<uint64_t> event_create_calls{ 0 };
    std::atomic<uint64_t> event_record_calls{ 0 };
    std::atomic<uint64_t> event_synchronize_calls{ 0 };
    std::atomic<uint64_t> event_elapsed_calls{ 0 };
    std::atomic<uint64_t> event_destroy_calls{ 0 };
    std::atomic<uint64_t> graph_measurements{ 0 };
    std::atomic<uint64_t> last_graph_micros{ 0 };
};

static DeviceTimingCounters g_counters;

#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
struct iree_dynamic_library_t;

using hipCtx_t    = void *;
using hipStream_t = void *;
using hipEvent_t  = void *;
using hipGraphNode_t = void *;
using hipGraph_t  = void *;
using hipError_t  = int;
using hipDevice_t = int;
using hipDeviceAttribute_t = int;
using hipFuncAttribute = int;

struct hipUUID {
    char bytes[16];
};

struct HIP_MEMCPY3D;
struct hipDeviceProp_tR0000;

static constexpr hipError_t hipSuccess      = 0;
static constexpr unsigned   hipEventDefault = 0;

struct HrxHipDynamicSymbols {
    iree_dynamic_library_t * dylib;
    hipError_t (*hipCtxSetCurrent)(hipCtx_t);
    hipError_t (*hipCtxGetCurrent)(hipCtx_t *);
    hipError_t (*hipCtxPushCurrent)(hipCtx_t);
    hipError_t (*hipCtxPopCurrent)(hipCtx_t *);
    hipError_t (*hipDeviceCanAccessPeer)(int *, int, int);
    hipError_t (*hipDeviceEnablePeerAccess)(int, unsigned int);
    hipError_t (*hipDeviceGet)(hipDevice_t *, int);
    hipError_t (*hipDeviceGetAttribute)(int *, hipDeviceAttribute_t, int);
    hipError_t (*hipDeviceGetName)(char *, int, hipDevice_t);
    hipError_t (*hipDeviceGetUuid)(hipUUID *, hipDevice_t);
    hipError_t (*hipDevicePrimaryCtxRelease)(hipDevice_t);
    hipError_t (*hipDevicePrimaryCtxRetain)(hipCtx_t *, hipDevice_t);
    hipError_t (*hipDrvGraphAddMemcpyNode)(hipGraphNode_t *, hipGraph_t, const hipGraphNode_t *, size_t,
                                           const HIP_MEMCPY3D *, hipCtx_t);
    hipError_t (*hipEventCreate)(hipEvent_t *);
    hipError_t (*hipEventCreateWithFlags)(hipEvent_t *, unsigned int);
    hipError_t (*hipEventDestroy)(hipEvent_t);
    hipError_t (*hipEventElapsedTime)(float *, hipEvent_t, hipEvent_t);
    hipError_t (*hipEventQuery)(hipEvent_t);
    hipError_t (*hipEventRecord)(hipEvent_t, hipStream_t);
    hipError_t (*hipEventSynchronize)(hipEvent_t);
    hipError_t (*hipFree)(void *);
    hipError_t (*hipFreeAsync)(void *, hipStream_t);
    hipError_t (*hipFuncGetAttribute)(int *, hipFuncAttribute, const void *);
    hipError_t (*hipFuncSetAttribute)(const void *, hipFuncAttribute, int);
    hipError_t (*hipGetDeviceCount)(int *);
    hipError_t (*hipGetDeviceProperties)(hipDeviceProp_tR0000 *, int);
    const char * (*hipGetErrorName)(hipError_t);
    const char * (*hipGetErrorString)(hipError_t);
};

struct HrxHipPerDeviceInfo {
    hipCtx_t    hip_context;
    hipDevice_t hip_device;
    hipStream_t hip_dispatch_stream;
};

struct HipDeviceLayout {
    iree_hal_resource_t              resource;
    iree_string_view_t               identifier;
    iree_arena_block_pool_t          block_pool;
    void *                           driver;
    const HrxHipDynamicSymbols *     hip_symbols;
    const void *                     nccl_symbols;
    iree_hal_hip_device_params_t     params;
    iree_allocator_t                 host_allocator;
    void *                           proactor_pool;
    void *                           proactor;
    iree_hal_device_event_sink_t     event_sink;
    void *                           frontier_tracker;
    iree_async_axis_t                axis;
    iree_atomic_int64_t              epoch;
    bool                             supports_memory_pools;
    iree_hal_allocator_t *           device_allocator;
    bool                             uses_external_stream;
    void *                           channel_provider;
    iree_hal_device_spec_t *         device_spec;
    iree_hal_device_topology_info_t  topology_info;
    void *                           cleanup_thread;
    void *                           buffer_free_thread;
    iree_host_size_t                 device_count;
    HrxHipPerDeviceInfo              devices[];
};

static std::string hip_error_message(const HrxHipDynamicSymbols * symbols, hipError_t error) {
    const char * name    = symbols != nullptr && symbols->hipGetErrorName != nullptr ? symbols->hipGetErrorName(error) : nullptr;
    const char * details = symbols != nullptr && symbols->hipGetErrorString != nullptr ? symbols->hipGetErrorString(error) : nullptr;
    std::string message  = name != nullptr ? name : "hipErrorUnknown";
    if (details != nullptr && details[0] != '\0') {
        message += ": ";
        message += details;
    }
    return message;
}

static std::string hrx_status_message(hrx_status_t status) {
    if (hrx_status_is_ok(status)) {
        return {};
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    std::string result = message != nullptr ? std::string(message, length) : "unknown HRX error";
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return result;
}

class HipContextScope {
  public:
    HipContextScope(const HrxHipDynamicSymbols * symbols, hipCtx_t target) : symbols_(symbols) {
        if (symbols_ == nullptr || target == nullptr) {
            return;
        }
        if (symbols_->hipCtxGetCurrent(&previous_) != hipSuccess) {
            previous_ = nullptr;
        }
        active_ = symbols_->hipCtxSetCurrent(target) == hipSuccess;
    }

    ~HipContextScope() {
        if (active_ && symbols_ != nullptr) {
            (void) symbols_->hipCtxSetCurrent(previous_);
        }
    }

    bool active() const { return active_; }

  private:
    const HrxHipDynamicSymbols * symbols_  = nullptr;
    hipCtx_t                    previous_ = nullptr;
    bool                        active_   = false;
};
#endif

}  // namespace

bool device_timing_environment_enabled() {
    return environment_flag_enabled("HRX_ENABLE_DEVICE_TIMING");
}

bool device_timing_internals_available() {
#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    return true;
#else
    return false;
#endif
}

void reset_device_timing_test_snapshot() {
    g_counters.init_attempts.store(0, std::memory_order_relaxed);
    g_counters.event_create_calls.store(0, std::memory_order_relaxed);
    g_counters.event_record_calls.store(0, std::memory_order_relaxed);
    g_counters.event_synchronize_calls.store(0, std::memory_order_relaxed);
    g_counters.event_elapsed_calls.store(0, std::memory_order_relaxed);
    g_counters.event_destroy_calls.store(0, std::memory_order_relaxed);
    g_counters.graph_measurements.store(0, std::memory_order_relaxed);
    g_counters.last_graph_micros.store(0, std::memory_order_relaxed);
}

DeviceTimingTestSnapshot device_timing_test_snapshot() {
    DeviceTimingTestSnapshot snapshot;
    snapshot.env_enabled            = device_timing_environment_enabled();
    snapshot.internals_available    = device_timing_internals_available();
    snapshot.init_attempts          = g_counters.init_attempts.load(std::memory_order_relaxed);
    snapshot.event_create_calls     = g_counters.event_create_calls.load(std::memory_order_relaxed);
    snapshot.event_record_calls     = g_counters.event_record_calls.load(std::memory_order_relaxed);
    snapshot.event_synchronize_calls = g_counters.event_synchronize_calls.load(std::memory_order_relaxed);
    snapshot.event_elapsed_calls    = g_counters.event_elapsed_calls.load(std::memory_order_relaxed);
    snapshot.event_destroy_calls    = g_counters.event_destroy_calls.load(std::memory_order_relaxed);
    snapshot.graph_measurements     = g_counters.graph_measurements.load(std::memory_order_relaxed);
    snapshot.last_graph_ms          = static_cast<double>(g_counters.last_graph_micros.load(std::memory_order_relaxed)) / 1000.0;
    return snapshot;
}

void device_timing_test_probe_flag_off_path() {
    DeviceTimingManager manager;
    DeviceTimingManager::GraphMeasurement measurement = manager.begin_graph_measurement(nullptr);
    manager.cancel_graph_measurement(measurement);
}

DeviceTimingManager::~DeviceTimingManager() {
#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    const auto * symbols = static_cast<const HrxHipDynamicSymbols *>(hip_symbols_);
    if (!initialized_ || symbols == nullptr || hip_context_ == nullptr) {
        return;
    }
    HipContextScope scope(symbols, static_cast<hipCtx_t>(hip_context_));
    for (void * event : owned_events_) {
        if (event == nullptr) {
            continue;
        }
        (void) symbols->hipEventDestroy(static_cast<hipEvent_t>(event));
        g_counters.event_destroy_calls.fetch_add(1, std::memory_order_relaxed);
    }
#endif
}

void DeviceTimingManager::report_unavailable_once(const char * message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (unavailable_reported_) {
        return;
    }
    unavailable_reported_ = true;
    std::fprintf(stderr, "HRX device timing disabled: %s\n", message);
    std::fflush(stderr);
}

bool DeviceTimingManager::ensure_initialized(hrx_stream_t stream) {
    if (!device_timing_environment_enabled()) {
        return false;
    }
#ifndef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    (void)stream;
    report_unavailable_once("build lacks HIP driver internals needed to reach the dispatch stream");
    return false;
#else
    if (!using_hip_driver()) {
        report_unavailable_once("HRX_ENABLE_DEVICE_TIMING currently supports only HRX_GPU_DRIVER=hip");
        return false;
    }
    if (initialized_) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return true;
    }
    const auto fail_locked = [&](const char * message) {
        if (!unavailable_reported_) {
            unavailable_reported_ = true;
            std::fprintf(stderr, "HRX device timing disabled: %s\n", message);
            std::fflush(stderr);
        }
        return false;
    };
    if (!init_attempted_) {
        g_counters.init_attempts.fetch_add(1, std::memory_order_relaxed);
        init_attempted_ = true;
    }
    if (stream == nullptr) {
        return fail_locked("missing HRX stream");
    }

    hrx_device_t device = nullptr;
    hrx_status_t  stream_status = hrx_stream_get_device(stream, &device);
    if (!hrx_status_is_ok(stream_status) || device == nullptr) {
        const std::string error = hrx_status_message(stream_status);
        return fail_locked(error.empty() ? "failed to resolve HRX stream device" : error.c_str());
    }

    if (device->hal_device == nullptr) {
        return fail_locked("missing HIP HAL device");
    }

    auto * hip_device = reinterpret_cast<HipDeviceLayout *>(device->hal_device);
    if (device->ordinal < 0 || static_cast<iree_host_size_t>(device->ordinal) >= hip_device->device_count) {
        return fail_locked("HIP device ordinal is outside the HAL topology");
    }
    const HrxHipPerDeviceInfo & per_device = hip_device->devices[device->ordinal];
    if (hip_device->hip_symbols == nullptr || per_device.hip_context == nullptr || per_device.hip_dispatch_stream == nullptr) {
        return fail_locked("HIP dispatch stream is unavailable");
    }

    hip_symbols_ = hip_device->hip_symbols;
    hip_context_ = per_device.hip_context;
    hip_stream_  = per_device.hip_dispatch_stream;
    initialized_ = true;
    return true;
#endif
}

void * DeviceTimingManager::acquire_event() {
#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    const auto * symbols = static_cast<const HrxHipDynamicSymbols *>(hip_symbols_);
    if (symbols == nullptr || hip_context_ == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!available_events_.empty()) {
        void * event = available_events_.back();
        available_events_.pop_back();
        return event;
    }

    HipContextScope scope(symbols, static_cast<hipCtx_t>(hip_context_));
    if (!scope.active()) {
        return nullptr;
    }

    hipEvent_t event = nullptr;
    if (symbols->hipEventCreateWithFlags(&event, hipEventDefault) != hipSuccess) {
        return nullptr;
    }
    owned_events_.push_back(event);
    g_counters.event_create_calls.fetch_add(1, std::memory_order_relaxed);
    return event;
#else
    return nullptr;
#endif
}

void DeviceTimingManager::release_event(void * event) {
    if (event == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    available_events_.push_back(event);
}

DeviceTimingManager::GraphMeasurement DeviceTimingManager::begin_graph_measurement(hrx_stream_t stream) {
    GraphMeasurement measurement;
    if (!ensure_initialized(stream)) {
        return measurement;
    }
#ifdef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    const auto * symbols = static_cast<const HrxHipDynamicSymbols *>(hip_symbols_);
    void *       event   = acquire_event();
    if (symbols == nullptr || event == nullptr) {
        report_unavailable_once("failed to acquire HIP timing event");
        return measurement;
    }

    HipContextScope scope(symbols, static_cast<hipCtx_t>(hip_context_));
    if (!scope.active()) {
        release_event(event);
        report_unavailable_once("failed to enter the HIP dispatch context");
        return measurement;
    }
    const hipError_t record = symbols->hipEventRecord(static_cast<hipEvent_t>(event), static_cast<hipStream_t>(hip_stream_));
    if (record != hipSuccess) {
        release_event(event);
        report_unavailable_once(hip_error_message(symbols, record).c_str());
        return measurement;
    }
    g_counters.event_record_calls.fetch_add(1, std::memory_order_relaxed);
    measurement.start_event = event;
    measurement.active      = true;
#endif
    return measurement;
}

std::optional<double> DeviceTimingManager::finish_graph_measurement(GraphMeasurement & measurement) {
    if (!measurement.active) {
        return std::nullopt;
    }
#ifndef GGML_HRX_HAVE_DEVICE_TIMING_INTERNALS
    measurement = {};
    return std::nullopt;
#else
    const auto * symbols = static_cast<const HrxHipDynamicSymbols *>(hip_symbols_);
    void *       end_event = acquire_event();
    if (symbols == nullptr || end_event == nullptr) {
        cancel_graph_measurement(measurement);
        report_unavailable_once("failed to acquire HIP completion event");
        return std::nullopt;
    }

    HipContextScope scope(symbols, static_cast<hipCtx_t>(hip_context_));
    if (!scope.active()) {
        release_event(end_event);
        cancel_graph_measurement(measurement);
        report_unavailable_once("failed to enter the HIP dispatch context");
        return std::nullopt;
    }

    const hipError_t record = symbols->hipEventRecord(static_cast<hipEvent_t>(end_event), static_cast<hipStream_t>(hip_stream_));
    if (record != hipSuccess) {
        release_event(end_event);
        cancel_graph_measurement(measurement);
        report_unavailable_once(hip_error_message(symbols, record).c_str());
        return std::nullopt;
    }
    g_counters.event_record_calls.fetch_add(1, std::memory_order_relaxed);

    const hipError_t sync = symbols->hipEventSynchronize(static_cast<hipEvent_t>(end_event));
    if (sync != hipSuccess) {
        release_event(end_event);
        cancel_graph_measurement(measurement);
        report_unavailable_once(hip_error_message(symbols, sync).c_str());
        return std::nullopt;
    }
    g_counters.event_synchronize_calls.fetch_add(1, std::memory_order_relaxed);

    float elapsed_ms = 0.0f;
    const hipError_t elapsed = symbols->hipEventElapsedTime(&elapsed_ms, static_cast<hipEvent_t>(measurement.start_event),
                                                            static_cast<hipEvent_t>(end_event));
    if (elapsed != hipSuccess) {
        release_event(end_event);
        cancel_graph_measurement(measurement);
        report_unavailable_once(hip_error_message(symbols, elapsed).c_str());
        return std::nullopt;
    }
    g_counters.event_elapsed_calls.fetch_add(1, std::memory_order_relaxed);
    g_counters.graph_measurements.fetch_add(1, std::memory_order_relaxed);
    g_counters.last_graph_micros.store(static_cast<uint64_t>(elapsed_ms * 1000.0f), std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++graph_samples_;
        graph_total_ms_ += elapsed_ms;
        if (elapsed_ms > graph_max_ms_) {
            graph_max_ms_ = elapsed_ms;
        }
    }

    release_event(measurement.start_event);
    release_event(end_event);
    measurement = {};
    return static_cast<double>(elapsed_ms);
#endif
}

void DeviceTimingManager::cancel_graph_measurement(GraphMeasurement & measurement) {
    if (!measurement.active) {
        return;
    }
    release_event(measurement.start_event);
    measurement = {};
}

void DeviceTimingManager::print_shutdown_summary(const char * backend_name) const {
    if (!device_timing_environment_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (graph_samples_ == 0) {
        return;
    }
    std::fprintf(stderr,
                 "HRX device time (~0.5ms HIP event granularity): backend=%s graphs=%llu total=%.3fms avg=%.3fms max=%.3fms\n",
                 backend_name != nullptr ? backend_name : "HRX",
                 static_cast<unsigned long long>(graph_samples_), graph_total_ms_,
                 graph_total_ms_ / static_cast<double>(graph_samples_), graph_max_ms_);
    std::fflush(stderr);
}

}  // namespace ggml::hrx
