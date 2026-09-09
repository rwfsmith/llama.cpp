#pragma once

#include "hrx_runtime.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace ggml::hrx {

struct DeviceTimingTestSnapshot {
    bool     env_enabled          = false;
    bool     internals_available  = false;
    uint64_t init_attempts        = 0;
    uint64_t event_create_calls   = 0;
    uint64_t event_record_calls   = 0;
    uint64_t event_synchronize_calls = 0;
    uint64_t event_elapsed_calls  = 0;
    uint64_t event_destroy_calls  = 0;
    uint64_t graph_measurements   = 0;
    double   last_graph_ms        = 0.0;
};

bool                     device_timing_environment_enabled();
bool                     device_timing_internals_available();
void                     reset_device_timing_test_snapshot();
DeviceTimingTestSnapshot device_timing_test_snapshot();
void                     device_timing_test_probe_flag_off_path();

class DeviceTimingManager {
  public:
    struct GraphMeasurement {
        void * start_event = nullptr;
        bool   active      = false;
    };

    DeviceTimingManager() = default;
    ~DeviceTimingManager();

    GraphMeasurement       begin_graph_measurement(hrx_stream_t stream);
    std::optional<double>  finish_graph_measurement(GraphMeasurement & measurement);
    void                   cancel_graph_measurement(GraphMeasurement & measurement);
    void                   print_shutdown_summary(const char * backend_name) const;

    DeviceTimingManager(const DeviceTimingManager &)             = delete;
    DeviceTimingManager & operator=(const DeviceTimingManager &) = delete;

  private:
    bool  ensure_initialized(hrx_stream_t stream);
    void * acquire_event();
    void   release_event(void * event);
    void   report_unavailable_once(const char * message);

    mutable std::mutex mutex_;
    bool               init_attempted_        = false;
    bool               initialized_           = false;
    bool               unavailable_reported_  = false;
    std::vector<void *> available_events_;
    std::vector<void *> owned_events_;
    void *             hip_context_           = nullptr;
    void *             hip_stream_            = nullptr;
    const void *       hip_symbols_           = nullptr;
    uint64_t           graph_samples_         = 0;
    double             graph_total_ms_        = 0.0;
    double             graph_max_ms_          = 0.0;
};

}  // namespace ggml::hrx
