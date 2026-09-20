import json
import re
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from inferencia_pc import MODEL_PATH, prepare_input
from teste_inferencia import IMAGE_INDEX, get_test_sample

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PC_DIR = BASE_DIR / "results_pc"
RESOLVERS = {
    "builtin_without_default_delegates": (
        OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    ),
    "builtin_ref": OpResolverType.BUILTIN_REF,
}
TRACE_RE = re.compile(
    r"OPTRACE op=(\d+) name=([^ ]+) tensor=(\d+) bytes=(\d+) "
    r"hash=(?:0x)?([0-9a-fA-F]+) min=(-?\d+) max=(-?\d+) sum=(-?\d+)"
)


def fnv1a32(data):
    value = 2166136261
    for byte in memoryview(np.ascontiguousarray(data)).cast("B"):
        value = ((value ^ int(byte)) * 16777619) & 0xFFFFFFFF
    return value


def tensor_stats(op_index, op_name, tensor_index, tensor):
    raw = np.ascontiguousarray(tensor)
    flat = raw.reshape(-1)
    return {
        "op": int(op_index),
        "name": op_name,
        "tensor": int(tensor_index),
        "bytes": int(raw.nbytes),
        "hash": f"{fnv1a32(raw):08x}",
        "min": int(flat.min()) if flat.size else 0,
        "max": int(flat.max()) if flat.size else 0,
        "sum": int(flat.astype(np.int64).sum()),
    }


def run_pc_trace(resolver_type):
    image, _, image_path = get_test_sample(IMAGE_INDEX)
    quantized = prepare_input(image)
    print(f"Imagem {image_path}: input_fnv1a={fnv1a32(quantized):08x}")
    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        experimental_op_resolver_type=resolver_type,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], quantized[None])
    interpreter.invoke()

    records = []
    for op in interpreter._get_ops_details():
        for tensor_index in op["outputs"]:
            tensor_index = int(tensor_index)
            if tensor_index < 0:
                continue
            tensor = interpreter.get_tensor(tensor_index)
            records.append(
                tensor_stats(op["index"], op["op_name"], tensor_index, tensor)
            )
    return records


def parse_device_trace(path):
    records = []
    for match in TRACE_RE.finditer(Path(path).read_text(errors="replace")):
        op, name, tensor, size, hash_value, min_value, max_value, sum_value = match.groups()
        records.append(
            {
                "op": int(op),
                "name": name,
                "tensor": int(tensor),
                "bytes": int(size),
                "hash": f"{int(hash_value, 16):08x}",
                "min": int(min_value),
                "max": int(max_value),
                "sum": int(sum_value),
            }
        )
    return records


def records_by_key(records, label):
    indexed = {(record["op"], record["tensor"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"Trace {label} contém registros repetidos")
    return indexed


def compare_records(left_name, left_records, right_name, right_records):
    left_by_key = records_by_key(left_records, left_name)
    right_by_key = records_by_key(right_records, right_name)
    missing = left_by_key.keys() - right_by_key.keys()
    extra = right_by_key.keys() - left_by_key.keys()
    if missing or extra:
        raise ValueError(
            f"Traces incompatíveis: ausentes em {right_name}={sorted(missing)}, "
            f"extras em {right_name}={sorted(extra)}"
        )

    print(
        f"\ncomparação={left_name} x {right_name} "
        f"registros={len(left_records)}"
    )
    first_hash_difference = None
    first_stats_difference = None
    for left in left_records:
        right = right_by_key[(left["op"], left["tensor"])]
        if left["hash"] != right["hash"] and first_hash_difference is None:
            first_hash_difference = (left, right)
        stats_equal = all(
            left[key] == right[key]
            for key in ("name", "bytes", "min", "max", "sum")
        )
        if not stats_equal and first_stats_difference is None:
            first_stats_difference = (left, right)

    for label, difference in (
        ("primeira diferença de hash", first_hash_difference),
        ("primeira diferença estatística", first_stats_difference),
    ):
        print(f"\n{label}:")
        if difference is None:
            print("nenhuma")
        else:
            left, right = difference
            print(f"{left_name}:", left)
            print(f"{right_name}:", right)

    return first_hash_difference is not None or first_stats_difference is not None


def save_trace(stem, resolver_name, records):
    output_path = RESULTS_PC_DIR / f"{stem}_optrace_{resolver_name}.json"
    output_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"trace_pc_path: {output_path}")
    for record in records:
        values = " ".join(
            f"{key}={record[key]}"
            for key in ("op", "name", "tensor", "bytes", "hash", "min", "max", "sum")
        )
        print(f"OPTRACE_PC resolver={resolver_name} {values}")


def main():
    RESULTS_PC_DIR.mkdir(parents=True, exist_ok=True)
    _, _, image_path = get_test_sample(IMAGE_INDEX)
    traces = {}
    for resolver_name, resolver_type in RESOLVERS.items():
        print(f"\nExecutando resolver: {resolver_name}")
        records = run_pc_trace(resolver_type)
        traces[resolver_name] = records
        save_trace(image_path.stem, resolver_name, records)

    differences = compare_records(
        "builtin_without_default_delegates",
        traces["builtin_without_default_delegates"],
        "builtin_ref",
        traces["builtin_ref"],
    )

    if len(sys.argv) >= 2:
        device_records = parse_device_trace(sys.argv[1])
        if not device_records:
            raise ValueError("Log sem registros OPTRACE; comparação não realizada")
        for resolver_name, records in traces.items():
            differences |= compare_records(
                resolver_name, records, "esp32", device_records
            )
    else:
        print("\nLog ESP32 não informado; comparação limitada aos resolvers do PC.")

    if differences:
        raise SystemExit(1)
    print("Todos os registros dos traces coincidem.")


if __name__ == "__main__":
    main()
