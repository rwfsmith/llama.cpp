#!/usr/bin/env python3

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Tuple


DEFAULT_KERNEL_FAMILY = "qwen3_moe"

SOURCE_ARRAY_TEMPLATE = """static const unsigned char {symbol}[] = {{
{bytes}
}};
static constexpr size_t {symbol}Size = {size};
"""

DEPENDENCY_TABLE_TEMPLATE = """static const KernelSourceSpan {dependency_table}[] = {{
{dependencies}
}};
"""

SOURCE_RECORD_TEMPLATE = """static const KernelSource {record} = {{
    {{ reinterpret_cast<const char *>({source_symbol}), {source_symbol}Size, {source_format} }},
    {dependency_table},
    {dependency_count},
}};
"""

CORPUS_ARRAY_TEMPLATE = """static {type} {symbol}[] = {{
{values}
}};
"""

KERNEL_RECORD_TEMPLATE = """    {{
        {family},
        {name},
        kernel_catalog_id({family}, {name}),
        {source},
        {dependencies},
        {symbol},
        "amdgpu",
        {target_selector},
        {{ nullptr, 0 }},
        {scalar_parameters},
        {bindings},
        {source_digest},
        {workload_parameters},
        {launch_parameters},
        {{
            {compile_mode},
            {link_module},
            {primary_sources},
            {library_sources},
        }},
    }},"""

SOURCE_DATA_TEMPLATE = """{source_arrays}
{dependency_tables}
{source_records}
static const KernelSourceRecordEntry kKernelSourceRecords[] = {{
{lookup_entries}
}};
"""

CORPUS_DATA_TEMPLATE = """{kernel_arrays}
static const KernelDefinition kQwenKernelDefinitions[] = {{
{kernel_records}
}};

static const KernelCorpus kQwenKernelCorpus = {{
    "ggml-hrx-kernel-corpus-v2",
    {upstream_revision},
    {corpus_digest},
    {recipe_digest},
    {plan_case_count},
    {{ kQwenKernelDefinitions, {kernel_count} }},
}};
"""

CATALOG_DATA_TEMPLATE = """struct KernelCatalogEntry {{
    const char * family;
    const char * name;
}};

static constexpr KernelCatalogEntry kKernelCatalogEntries[] = {{
{kernel_entries}
}};

constexpr bool kernel_catalog_entry_exists(const char * family, const char * name) {{
    for (const KernelCatalogEntry & known : kKernelCatalogEntries) {{
        if (kernel_catalog_name_equal(family, known.family) && kernel_catalog_name_equal(name, known.name)) {{
            return true;
        }}
    }}
    return false;
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embedded kernel corpus data includes.")
    parser.add_argument("--source-output", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-output", type=pathlib.Path, required=True)
    parser.add_argument("--catalog-output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-format", choices=("text", "binary"), default="text")
    parser.add_argument("--loom-link", type=pathlib.Path)
    parser.add_argument("--loom-format", type=pathlib.Path)
    parser.add_argument("--depfile", type=pathlib.Path)
    return parser.parse_args()


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def read_bytes(path: pathlib.Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def run_tool(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def require_tool(command: List[str]) -> None:
    result = run_tool(command)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")


def convert_source_to_bytecode(source_path: pathlib.Path, loom_link: pathlib.Path, loom_format: pathlib.Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ggml-hrx-loom-") as temp_dir_name:
        temp_dir = pathlib.Path(temp_dir_name)
        stripped = temp_dir / "stripped.loom"
        bytecode = temp_dir / "stripped.loombc"
        require_tool([
            str(loom_link),
            "--verify=false",
            "--mode=archive",
            "--strip-check",
            "--to=text",
            f"--output={stripped}",
            str(source_path),
        ])
        format_result = run_tool([
            str(loom_format),
            "--from=text",
            "--to=bc",
            f"--output={bytecode}",
            str(stripped),
        ])
        if format_result.returncode:
            require_tool([
                str(loom_link),
                "--verify=false",
                "--mode=archive",
                "--strip-check",
                "--to=bc",
                f"--output={bytecode}",
                str(source_path),
            ])
        return read_bytes(bytecode)


def sanitize_symbol(path: str, index: int) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path)
    if not stem or stem[0].isdigit():
        stem = f"_{stem}"
    return f"kernel_source_{index}_{stem}"


def format_byte_array(data: bytes) -> str:
    if not data:
        return ""
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines) + "\n"


def escape_cpp_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def cpp_string(text: str) -> str:
    return f"\"{escape_cpp_string(text)}\""


def resource_access_value(access: str) -> str:
    if access == "read":
        return "ResourceAccess::Read"
    if access == "write":
        return "ResourceAccess::Write"
    if access == "read_write":
        return "ResourceAccess::ReadWrite"
    raise RuntimeError(f"invalid kernel binding access metadata: {access}")


def span_initializer(symbol: str, count: int) -> str:
    if count == 0:
        return "{ nullptr, 0 }"
    return "{ " + symbol + ", " + str(count) + " }"


def typed_array(symbol: str, value_type: str, values: List[str]) -> Tuple[str, str]:
    if not values:
        return "", "{ nullptr, 0 }"
    array = CORPUS_ARRAY_TEMPLATE.format(
        type=value_type,
        symbol=symbol,
        values="\n".join("    " + value + "," for value in values),
    )
    return array, span_initializer(symbol, len(values))


def string_array(symbol: str, items: Iterable[str]) -> Tuple[str, str]:
    return typed_array(symbol, "const char * const", [cpp_string(item) for item in items])


def source_ref_array(symbol: str, items: Iterable[str], source_records: Dict[str, str]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item) + ", &" + source_records[item] + " }"
        for item in items
    ]
    return typed_array(symbol, "const KernelSourceRef", values)


def scalar_array(symbol: str, items: Iterable[dict]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item["name"]) + ", " + cpp_string(item["type"]) + " }"
        for item in items
    ]
    return typed_array(symbol, "const KernelScalarDefinition", values)


def binding_array(symbol: str, names: List[str], access: List[str]) -> Tuple[str, str]:
    if len(names) != len(access):
        raise RuntimeError("kernel binding access metadata has the wrong arity")
    values = [
        "{ " + cpp_string(name) + ", " + resource_access_value(access_value) + " }"
        for name, access_value in zip(names, access)
    ]
    return typed_array(symbol, "const KernelBindingDefinition", values)


def scalar_parameter_names(workload_parameters: List[dict], launch_parameters: List[dict]) -> List[str]:
    names: List[str] = []
    for parameter in [*workload_parameters, *launch_parameters]:
        name = parameter["name"]
        if name not in names:
            names.append(name)
    return names


def depfile_escape(path: pathlib.Path) -> str:
    return str(path).replace("\\", "\\\\").replace(" ", "\\ ")


def collect_sources(manifest: dict) -> Tuple[List[str], Dict[str, List[str]]]:
    source_dependencies: Dict[str, List[str]] = {}
    all_sources = set()

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        primary_sources = recipe.get("primary_sources", [])
        library_sources = recipe.get("library_sources", [])
        if len(primary_sources) != 1:
            raise RuntimeError("expected each export compile_recipe to have exactly one primary source")

        primary = primary_sources[0]
        dependencies = list(library_sources)
        previous = source_dependencies.get(primary)
        if previous is not None and previous != dependencies:
            raise RuntimeError(f"conflicting dependency list for {primary}")
        source_dependencies[primary] = dependencies
        all_sources.add(primary)
        all_sources.update(dependencies)

    return sorted(all_sources), source_dependencies


def manifest_file_digests(manifest: dict) -> Dict[str, str]:
    return {file["path"]: file["sha256"] for file in manifest.get("files", [])}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def with_owned_kernels(manifest: dict, corpus_dir: pathlib.Path) -> dict:
    # Owned sources are registered here, without changing pinned vendor provenance.
    iq_parameters = ["input_size", "output_size", "expert_count", "route_count",
                     "token_count", "route_id_stride"]
    iq_bindings = ["weight", "input", "ids", "tables", "output"]
    iq_access = ["read", "read", "read", "read", "write"]
    owned_sources = [
        ("../hrx_owned/qwen4exp_norm.loom", [
            ("ggml_qsa_norm_f32", ["width", "heads", "tokens", "source_stride"],
             ["input", "output"], ["read", "write"]),
        ]),
        ("../hrx_owned/qwen4exp_rope.loom", [
            ("ggml_qsa_rope_f32", ["width", "heads", "tokens"],
             ["input", "positions", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/qsa_bf16_projection.loom", [
            ("ggml_qsa_bf16_projection", ["output_size", "token_count"],
             ["weight", "input", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/recurrent_concat_f32.loom", [
            ("ggml_recurrent_concat_f32", ["history_count", "token_count", "input_transposed"],
             ["history", "input", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/ple_conv_f32.loom", [
            ("ggml_ple_conv_f32", ["token_count"],
             ["activation0", "weight0", "activation1", "weight1",
              "activation2", "weight2", "activation3", "weight3", "output"],
             ["read", "read", "read", "read", "read", "read", "read", "read", "write"]),
        ]),
        ("../hrx_owned/swiglu_split_f32.loom", [
            ("ggml_swiglu_split_f32", ["element_count"],
             ["gate", "up", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/get_rows_f32.loom", [
            ("ggml_get_rows_f32", ["hidden_size", "vocabulary_count", "token_count"],
             ["source", "ids", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/qsa_get_rows_f16.loom", [
            ("ggml_qsa_get_rows_f16", ["hidden_size", "vocabulary_count", "token_count"],
             ["source", "ids", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/token_embedding_q4_k.loom", [
            ("ggml_token_embedding_q4_k_f32", ["hidden_size", "vocabulary_count", "token_count"],
             ["weight", "ids", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/token_embedding_q8_0.loom", [
            ("ggml_token_embedding_q8_0_f32", ["hidden_size", "vocabulary_count", "token_count"],
             ["weight", "ids", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/set_rows_f32.loom", [
            ("ggml_set_rows_f32",
             ["row_length", "row_count", "cache_rows", "source_stride", "output_stride",
              "index_stride", "output_f16", "index_i64"],
             ["source", "ids", "output"], ["read", "read", "read_write"]),
        ]),
        ("../hrx_owned/dense_q8_0_gemv.loom", [
            ("ggml_dense_q8_0_gemv_f32", ["input_size", "output_size"],
             ["input", "weight", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/dense_q8_narrow.loom", [
            ("ggml_q8_narrow_pack", ["input_size"],
             ["input", "packed"], ["read", "write"]),
            ("ggml_q8_narrow_dot", ["input_size", "output_size"],
             ["packed", "weight", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/dense_q8_narrow_batched.loom", [
            ("ggml_q8_narrow_pack_batched", ["input_size", "token_count"],
             ["input", "packed"], ["read", "write"]),
            ("ggml_q8_narrow_dot_batched", ["input_size", "output_size", "token_count"],
             ["packed", "weight", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/dense_quantized_f32_accum.loom", [
            (name, ["token_count"], ["input", "weight", "output"], ["read", "read", "write"])
            for name in ("ggml_dense_q4k_f32_accum", "ggml_dense_q6k_f32_accum",
                         "ggml_dense_q8_0_f32_accum", "ggml_dense_q8_0_rounded_gemv",
                         "ggml_dense_q6k_rounded_gemv")
        ]),
        ("../hrx_owned/qsa_mask.loom", [
            ("ggml_qsa_mask_fill", ["element_count", "element_bytes", "value_bits"],
             ["output"], ["write"]),
            ("ggml_qsa_mask_cast", ["element_count", "source_f16"],
             ["input", "output"], ["read", "write"]),
            ("ggml_qsa_mask_add_f16", ["element_count", "rhs_count"],
             ["lhs", "rhs", "output"], ["read", "read", "write"]),
        ]),
        ("../hrx_owned/mul_mat_id_iq.loom", [
            ("ggml_mul_mat_id_iq3_xxs_f32", iq_parameters, iq_bindings, iq_access),
            ("ggml_mul_mat_id_iq4_xs_f32", iq_parameters, iq_bindings, iq_access),
            ("ggml_mul_mat_id_iq3_xxs_f32_packet4", iq_parameters, iq_bindings, iq_access),
            ("ggml_mul_mat_id_iq4_xs_f32_packet4", iq_parameters, iq_bindings, iq_access),
        ]),
        ("../hrx_owned/moe_small_down.loom", [
            (name, ["token_count", "input_size", "route_count", "route_id_stride",
                    "expert_count", "output_size"],
             ["input", "ids", "route_weights", "weight", "output"],
             ["read", "read", "read", "read", "read_write"])
            for name in ("ggml_moe_small_down_q8_0", "ggml_moe_small_down_iq4_nl")
        ]),
    ]
    files = list(manifest["files"])
    exports = list(manifest["exports"])
    corpus_digest = manifest["corpus_sha256"]
    for source, definitions in owned_sources:
        dependencies = []
        if source == "../hrx_owned/moe_small_down.loom":
            dependencies = ["ggml/quantize_q8_1_x4.loom",
                            "qwen3_moe/routed_down_q8_0.loom",
                            "qwen3_moe/routed_down_iq4_nl.loom"]
        elif source == "../hrx_owned/dense_quantized_f32_accum.loom":
            dependencies = ["qwen3_moe/dense_linear_quantized_f16_wmma.loom",
                            "ggml/linear_q6k_q8_1_x4.loom"]
        digest = sha256(read_bytes(corpus_dir / source))
        files.append({"path": source, "sha256": digest})
        corpus_digest = sha256((corpus_digest + source + digest).encode("utf-8"))
        for name, parameter_names, bindings, access in definitions:
            parameters = [{"name": parameter, "type": "index"} for parameter in parameter_names]
            exports.append({
                "family": "hrx_owned",
                "name": name,
                "symbol": name,
                "source": source,
                "target_selector": "",
                "workload_parameters": parameters,
                "launch_parameters": parameters,
                "bindings": bindings,
                "binding_access": access,
                "compile_dependencies": dependencies,
                "compile_recipe": {
                    "mode": "direct",
                    "primary_sources": [source],
                    "library_sources": dependencies,
                },
            })
    return {
        **manifest,
        "files": files,
        "exports": exports,
        "corpus_sha256": corpus_digest,
    }


def kernel_catalog_id(family: str, name: str) -> int:
    hash_value = 1469598103934665603
    for byte in family.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    hash_value ^= 0
    hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for byte in name.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def with_qsa_launch_only_parameters(manifest: dict, corpus_dir: pathlib.Path) -> dict:
    # The pinned catalog incorrectly lists token_count as a root specialization
    # parameter. Both kernels have kernel.def @symbol(), with indices ONLY in
    # launch(...). Keep those runtime scalar arguments; do not edit vendor files.
    launch_only = {
        "qwen38_attention_decode_split_reduce": ["token_count", "token_ordinal"],
        "qwen38_attention_decode_split_wmma_partial": ["token_count", "token_ordinal", "control_position_index"],
    }
    exports = []
    corrections = []
    for export in manifest["exports"]:
        names = launch_only.get(export["name"]) if export.get("family") == "qwen4exp" else None
        if names is not None:
            source = read_text(corpus_dir / export["source"])
            root = r"kernel\.def\b[^\n]*@" + re.escape(export["symbol"]) + r"\s*\(\s*\)\s*\{"
            parameters = [{"name": name, "type": "index"} for name in names]
            if not re.search(root, source) or export["launch_parameters"] != parameters:
                raise RuntimeError(f"QSA launch-only ABI changed: {export['name']}")
            if export["workload_parameters"]:
                export = {**export, "workload_parameters": []}
                corrections.append({
                    "family": export["family"],
                    "name": export["name"],
                    "workload_parameters": [],
                    "launch_parameters": parameters,
                })
        exports.append(export)
    if not corrections:
        return manifest
    correction_digest = json.dumps(corrections, sort_keys=True, separators=(",", ":"))
    return {
        **manifest,
        "exports": exports,
        "corpus_sha256": sha256((manifest["corpus_sha256"] + correction_digest).encode("utf-8")),
    }


def generate_corpus_records(manifest: dict, source_records: Dict[str, str]) -> Tuple[str, str, int]:
    digests = manifest_file_digests(manifest)
    arrays = []
    records = []
    exports = manifest.get("exports", [])
    for index, export in enumerate(exports):
        recipe = export["compile_recipe"]
        dependencies = list(export["compile_dependencies"])
        library_sources = list(recipe["library_sources"])
        if dependencies != library_sources:
            raise RuntimeError("legacy dependency closure disagrees with compile recipe")
        workload_parameters = list(export["workload_parameters"])
        launch_parameters = list(export["launch_parameters"])

        dependencies_array, dependencies_span = string_array(f"kKernelDependencies{index}", dependencies)
        scalar_array_text, scalar_span = string_array(
            f"kKernelScalarParameters{index}",
            scalar_parameter_names(workload_parameters, launch_parameters),
        )
        bindings_array, bindings_span = binding_array(
            f"kKernelBindings{index}",
            list(export["bindings"]),
            list(export.get("binding_access", [])),
        )
        workload_array, workload_span = scalar_array(f"kKernelWorkloadParameters{index}", workload_parameters)
        launch_array, launch_span = scalar_array(f"kKernelLaunchParameters{index}", launch_parameters)
        primary_array, primary_span = source_ref_array(f"kKernelPrimarySources{index}", recipe["primary_sources"], source_records)
        library_array, library_span = source_ref_array(f"kKernelLibrarySources{index}", library_sources, source_records)
        arrays.extend(
            item for item in [
                dependencies_array,
                scalar_array_text,
                bindings_array,
                workload_array,
                launch_array,
                primary_array,
                library_array,
            ] if item
        )
        records.append(
            KERNEL_RECORD_TEMPLATE.format(
                family=cpp_string(export.get("family", DEFAULT_KERNEL_FAMILY)),
                name=cpp_string(export["name"]),
                symbol=cpp_string(export["symbol"]),
                target_selector=cpp_string(export.get("target_selector", "")),
                source=cpp_string(export["source"]),
                dependencies=dependencies_span,
                source_digest=cpp_string(digests[export["source"]]),
                scalar_parameters=scalar_span,
                bindings=bindings_span,
                workload_parameters=workload_span,
                launch_parameters=launch_span,
                compile_mode=cpp_string(recipe["mode"]),
                link_module=cpp_string(recipe.get("link_module", "")),
                primary_sources=primary_span,
                library_sources=library_span,
            )
        )
    return "\n".join(arrays), "\n".join(records), len(exports)


def generate_catalog_verifier(manifest: dict) -> str:
    kernel_entries = sorted(set(
        (export.get("family", DEFAULT_KERNEL_FAMILY), export["name"])
        for export in manifest.get("exports", [])
    ))
    ids: Dict[int, Tuple[str, str]] = {}
    for family, kernel_name in kernel_entries:
        catalog_id = kernel_catalog_id(family, kernel_name)
        if catalog_id in ids:
            previous_family, previous_name = ids[catalog_id]
            raise RuntimeError(
                "kernel catalog id collision: "
                f"{previous_family}/{previous_name} and {family}/{kernel_name}")
        ids[catalog_id] = (family, kernel_name)
    return CATALOG_DATA_TEMPLATE.format(
        kernel_entries="\n".join(
            "    { " + cpp_string(family) + ", " + cpp_string(kernel_name) + " },"
            for family, kernel_name in kernel_entries
        ),
    )


def generate_includes(args: argparse.Namespace, manifest: dict) -> Tuple[str, str, str, List[pathlib.Path], int]:
    if args.source_format == "binary" and (args.loom_link is None or args.loom_format is None):
        raise RuntimeError("binary source format requires --loom-link and --loom-format")

    corpus_dir = args.corpus_dir
    manifest = with_owned_kernels(manifest, corpus_dir)
    manifest = with_qsa_launch_only_parameters(manifest, corpus_dir)
    sources, source_dependencies = collect_sources(manifest)
    digests = manifest_file_digests(manifest)
    source_bytes: Dict[str, bytes] = {}
    source_format = "KERNEL_SOURCE_FORMAT_BINARY" if args.source_format == "binary" else "KERNEL_SOURCE_FORMAT_TEXT"
    input_files: List[pathlib.Path] = []

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        for primary in recipe.get("primary_sources", []):
            if primary not in sources:
                raise RuntimeError(f"export primary source is not embedded: {primary}")

    for source in sources:
        if source not in digests:
            raise RuntimeError(f"embedded source is missing from manifest file table: {source}")
        path = corpus_dir / source
        input_files.append(path)
        data = read_bytes(path)
        digest = sha256(data)
        if digest != digests[source]:
            raise RuntimeError(f"manifest digest mismatch for {source}: got {digest}, expected {digests[source]}")
        if args.source_format == "binary":
            data = convert_source_to_bytecode(path, args.loom_link, args.loom_format)
        source_bytes[source] = data

    source_symbols: Dict[str, str] = {}
    source_arrays = []
    for index, source in enumerate(sources):
        symbol = sanitize_symbol(source, index)
        source_symbols[source] = symbol
        data = source_bytes[source]
        source_arrays.append(
            SOURCE_ARRAY_TEMPLATE.format(
                symbol=symbol,
                bytes=format_byte_array(data),
                size=len(data),
            )
        )

    dependency_tables = []
    source_record_definitions = []
    source_record_symbols = {}
    lookup_entries = []
    for index, source in enumerate(sources):
        dependencies = source_dependencies.get(source, [])
        record = f"kernel_source_record_{index}"
        source_record_symbols[source] = record
        if dependencies:
            dependency_table = f"kernel_source_dependencies_{index}"
            entries = []
            for dependency in dependencies:
                symbol = source_symbols[dependency]
                entries.append(
                    f"    {{ reinterpret_cast<const char *>({symbol}), {symbol}Size, {source_format} }},"
                )
            dependency_tables.append(
                DEPENDENCY_TABLE_TEMPLATE.format(
                    dependency_table=dependency_table,
                    dependencies="\n".join(entries),
                )
            )
        else:
            dependency_table = "nullptr"

        source_record_definitions.append(
            SOURCE_RECORD_TEMPLATE.format(
                record=record,
                source_symbol=source_symbols[source],
                source_format=source_format,
                dependency_table=dependency_table,
                dependency_count=len(dependencies),
            )
        )
        lookup_entries.append(f"    {{ {cpp_string(source)}, &{record} }},")

    kernel_arrays, kernel_records, kernel_count = generate_corpus_records(manifest, source_record_symbols)
    return (
        SOURCE_DATA_TEMPLATE.format(
            source_arrays="\n".join(source_arrays),
            dependency_tables="\n".join(dependency_tables),
            source_records="\n".join(source_record_definitions),
            lookup_entries="\n".join(lookup_entries),
        ),
        CORPUS_DATA_TEMPLATE.format(
            kernel_arrays=kernel_arrays,
            kernel_records=kernel_records,
            upstream_revision=cpp_string(manifest["upstream_revision"]),
            corpus_digest=cpp_string(manifest["corpus_sha256"]),
            recipe_digest=cpp_string(manifest["build_bazel_sha256"]),
            plan_case_count=len(manifest["plan_cases"]),
            kernel_count=kernel_count,
        ),
        generate_catalog_verifier(manifest),
        input_files,
        sum(len(data) for data in source_bytes.values()),
    )


def write_depfile(path: pathlib.Path, outputs: List[pathlib.Path], inputs: List[pathlib.Path]) -> None:
    targets = " ".join(depfile_escape(output_path) for output_path in outputs)
    entries = [depfile_escape(input_path) for input_path in inputs]
    path.write_text(f"{targets}: {' '.join(entries)}\n")


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(read_text(args.manifest))
        source_include, corpus_include, catalog_include, input_files, byte_count = generate_includes(args, manifest)
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
        args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_text(source_include)
        args.corpus_output.write_text(corpus_include)
        args.catalog_output.write_text(catalog_include)
        if args.depfile is not None:
            args.depfile.parent.mkdir(parents=True, exist_ok=True)
            write_depfile(args.depfile, [args.source_output, args.corpus_output, args.catalog_output],
                          [args.manifest, *input_files])
    except Exception as exc:
        print(f"generate_kernel_corpus.py: {exc}", file=sys.stderr)
        return 1

    print(
        f"embedded kernel corpus: source_files={len(input_files)} source_bytes={byte_count} "
        f"source_format={args.source_format}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
