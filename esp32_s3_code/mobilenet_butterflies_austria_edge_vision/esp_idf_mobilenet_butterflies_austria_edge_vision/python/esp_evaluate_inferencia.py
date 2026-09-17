import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from ai_edge_litert.interpreter import Interpreter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from teste_inferencia import (
    CLASS_NAMES,
    TEST_DATASET_PATH,
    fnv1a32,
    get_test_samples,
    load_image,
    preprocess_image,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    BASE_DIR.parent / "data" / "model_int8_avgpool.tflite",
))
RESULTS_ROOT = Path(os.environ.get(
    "RESULTS_DIR",
    BASE_DIR / "results_esp" / "evaluation",
))
ESP32_IP = os.environ.get("ESP32_IP", "192.168.3.22")
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
STATUS_URL = f"http://{ESP32_IP}/status"
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "1"))

TIMING_FIELDS = (
    "preprocess_us",
    "host_roundtrip_us",
    "receive_us",
    "input_copy_us",
    "inference_us",
    "postprocess_us",
    "total_processing_us",
    "device_request_us",
    "host_overhead_us",
)
MEMORY_FIELDS = (
    "arena_used",
    "arena_capacity",
    "model_bytes",
    "internal_free",
    "psram_free",
)
PREDICTION_FIELDS = (
    "image_index",
    "image_path",
    "true_class",
    "true_name",
    "predicted_class",
    "predicted_name",
    "correct",
    "confidence",
    "input_fnv1a",
    "output_fnv1a",
    *TIMING_FIELDS,
    *MEMORY_FIELDS,
    "scores",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_metadata(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Esperado um tensor de entrada e um tensor de saída")

    input_info = inputs[0]
    output_info = outputs[0]
    input_shape = [int(value) for value in input_info["shape"]]
    output_shape = [int(value) for value in output_info["shape"]]
    input_scale, input_zero_point = input_info["quantization"]
    output_scale, output_zero_point = output_info["quantization"]

    if input_shape != [1, 3, 224, 224] or input_info["dtype"] != np.int8:
        raise ValueError("Entrada esperada: int8 [1,3,224,224]")
    if output_shape != [1, len(CLASS_NAMES)] or output_info["dtype"] != np.int8:
        raise ValueError(f"Saída esperada: int8 [1,{len(CLASS_NAMES)}]")
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("Parâmetros de quantização inválidos")

    preprocessing = {
        "shape": input_shape,
        "layout": "NCHW",
        "dtype": "int8",
        "scale": float(input_scale),
        "zero_point": int(input_zero_point),
    }
    return {
        "path": str(model_path.resolve()),
        "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "input": preprocessing,
        "output": {
            "shape": output_shape,
            "dtype": "int8",
            "scale": float(output_scale),
            "zero_point": int(output_zero_point),
        },
    }


def request_status(session):
    response = session.get(STATUS_URL, timeout=(5, REQUEST_TIMEOUT))
    response.raise_for_status()
    status = response.json()
    if not status.get("success") or not status.get("model_initialized"):
        raise RuntimeError("Modelo não inicializado no ESP32")
    return status


def validate_status(status, model_metadata):
    expected = model_metadata["input"]
    if status.get("input_shape") != expected["shape"]:
        raise RuntimeError("Shape de entrada do ESP32 difere do modelo local")
    if status.get("input_layout") != expected["layout"]:
        raise RuntimeError("Layout de entrada do ESP32 difere do modelo local")
    if status.get("input_dtype") != expected["dtype"]:
        raise RuntimeError("Tipo de entrada do ESP32 difere do modelo local")
    if status.get("input_zero_point") != expected["zero_point"]:
        raise RuntimeError("Zero-point do ESP32 difere do modelo local")
    if not np.isclose(
        status.get("input_scale"),
        expected["scale"],
        rtol=1e-7,
        atol=0,
    ):
        raise RuntimeError("Escala de entrada do ESP32 difere do modelo local")
    if status.get("output_classes") != len(CLASS_NAMES):
        raise RuntimeError("Quantidade de classes do ESP32 inválida")


def send_inference(session, quantized):
    started_ns = time.perf_counter_ns()
    response = session.post(
        PREDICT_URL,
        data=quantized.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=(5, REQUEST_TIMEOUT),
    )
    host_roundtrip_us = (time.perf_counter_ns() - started_ns) // 1000
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise RuntimeError(result.get("error_message", "Falha na inferência"))
    result["host_roundtrip_us"] = int(host_roundtrip_us)
    result["device_request_us"] = int(
        result["receive_us"] + result["total_processing_us"]
    )
    result["host_overhead_us"] = int(
        host_roundtrip_us - result["device_request_us"]
    )
    return result


def validate_result(result, quantized, model_metadata):
    expected_hash = f"{fnv1a32(quantized):08x}"
    if result.get("input_fnv1a") != expected_hash:
        raise RuntimeError("Hash da entrada divergiu entre PC e ESP32")
    if len(result.get("scores", [])) != len(CLASS_NAMES):
        raise RuntimeError("Quantidade de scores inválida")
    predicted = int(result["predicted_class"])
    if not 0 <= predicted < len(CLASS_NAMES):
        raise RuntimeError("Classe predita inválida")
    if result["model_bytes"] != model_metadata["bytes"]:
        raise RuntimeError("Tamanho do modelo no ESP32 difere do modelo local")


def describe(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Série de métricas vazia")
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "median": float(np.median(data)),
        "p05": float(np.percentile(data, 5)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def wilson_interval(correct, total, confidence_z=1.959963984540054):
    if total <= 0:
        raise ValueError("Total de amostras inválido")
    proportion = correct / total
    denominator = 1 + confidence_z ** 2 / total
    center = (
        proportion + confidence_z ** 2 / (2 * total)
    ) / denominator
    margin = confidence_z * math.sqrt(
        proportion * (1 - proportion) / total
        + confidence_z ** 2 / (4 * total ** 2)
    ) / denominator
    return [center - margin, center + margin]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def save_confusion_matrix(matrix, results_dir):
    with (results_dir / "confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["real/predita", *CLASS_NAMES])
        for class_name, values in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *values.tolist()])

    figure, axis = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Label predita")
    axis.set_ylabel("Label real")
    axis.set_title("Matriz de confusão - inferência ESP32-S3")
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)


def warmup(session, sample, preprocessing, model_metadata):
    image_path, _ = sample
    image = load_image(image_path)
    quantized = preprocess_image(image, preprocessing)
    for _ in tqdm(range(WARMUP_RUNS), desc="Warm-up"):
        result = send_inference(session, quantized)
        validate_result(result, quantized, model_metadata)


def evaluate(session, samples, preprocessing, model_metadata, results_dir):
    labels = []
    predictions = []
    rows = []
    csv_path = results_dir / "predictions.csv"
    jsonl_path = results_dir / "telemetry.jsonl"

    with csv_path.open("w", newline="") as csv_file, jsonl_path.open("w") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()

        progress = tqdm(samples, desc="Avaliando ESP32-S3")
        for image_index, (image_path, true_class) in enumerate(progress):
            preprocess_started_ns = time.perf_counter_ns()
            image = load_image(image_path)
            quantized = preprocess_image(image, preprocessing)
            preprocess_us = (
                time.perf_counter_ns() - preprocess_started_ns
            ) // 1000

            result = send_inference(session, quantized)
            validate_result(result, quantized, model_metadata)
            predicted_class = int(result["predicted_class"])
            labels.append(int(true_class))
            predictions.append(predicted_class)

            row = {
                "image_index": image_index,
                "image_path": str(image_path.relative_to(TEST_DATASET_PATH)),
                "true_class": int(true_class),
                "true_name": CLASS_NAMES[true_class],
                "predicted_class": predicted_class,
                "predicted_name": CLASS_NAMES[predicted_class],
                "correct": predicted_class == true_class,
                "confidence": float(result["confidence"]),
                "input_fnv1a": result["input_fnv1a"],
                "output_fnv1a": result["output_fnv1a"],
                "preprocess_us": int(preprocess_us),
                **{
                    field: int(result[field])
                    for field in TIMING_FIELDS
                    if field != "preprocess_us"
                },
                **{field: int(result[field]) for field in MEMORY_FIELDS},
                "scores": json.dumps(result["scores"], separators=(",", ":")),
            }
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()

            telemetry = {**row, "scores": result["scores"]}
            jsonl_file.write(json.dumps(telemetry, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            running_accuracy = float(np.mean(
                np.asarray(labels) == np.asarray(predictions)
            ))
            progress.set_postfix(
                accuracy=f"{running_accuracy:.4f}",
                inference_s=f"{result['inference_us'] / 1_000_000:.3f}",
            )

    return rows, labels, predictions


def build_summary(
    rows,
    labels,
    predictions,
    model_metadata,
    status_before,
    status_after,
    started_at,
    finished_at,
    run_wall_time_s,
    evaluation_wall_time_s,
):
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    correct = int(np.sum(labels_array == predictions_array))
    total = len(labels)
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "run_wall_time_s": run_wall_time_s,
        "evaluation_wall_time_s": evaluation_wall_time_s,
        "esp32_ip": ESP32_IP,
        "predict_url": PREDICT_URL,
        "dataset_path": str(TEST_DATASET_PATH.resolve()),
        "model": model_metadata,
        "device_status_before": status_before,
        "device_status_after": status_after,
        "warmup_runs": WARMUP_RUNS,
        "samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "accuracy_ci95_wilson": wilson_interval(correct, total),
        "classes": CLASS_NAMES,
        "timing_us": {
            field: describe([row[field] for row in rows])
            for field in TIMING_FIELDS
        },
        "memory_bytes": {
            field: describe([row[field] for row in rows])
            for field in MEMORY_FIELDS
        },
        "throughput_images_per_second": total / evaluation_wall_time_s,
    }


def save_classification_results(labels, predictions, results_dir):
    class_indices = list(range(len(CLASS_NAMES)))
    report_text = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_json = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    (results_dir / "classification_report.txt").write_text(report_text + "\n")
    save_json(results_dir / "classification_report.json", report_json)
    save_confusion_matrix(matrix, results_dir)
    return report_text


def main():
    if WARMUP_RUNS < 0:
        raise ValueError("WARMUP_RUNS deve ser maior ou igual a zero")

    model_metadata = load_model_metadata(MODEL_PATH)
    preprocessing = model_metadata["input"]
    samples = get_test_samples()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = RESULTS_ROOT / f"{timestamp}_{MODEL_PATH.stem}"
    results_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "status": "running",
        "started_at_utc": started_at,
        "esp32_ip": ESP32_IP,
        "dataset_path": str(TEST_DATASET_PATH.resolve()),
        "model": model_metadata,
        "warmup_runs": WARMUP_RUNS,
        "expected_samples": len(samples),
    }
    save_json(results_dir / "run.json", run_metadata)

    wall_started_ns = time.perf_counter_ns()
    try:
        with requests.Session() as session:
            status_before = request_status(session)
            validate_status(status_before, model_metadata)
            warmup(session, samples[0], preprocessing, model_metadata)
            evaluation_started_ns = time.perf_counter_ns()
            rows, labels, predictions = evaluate(
                session,
                samples,
                preprocessing,
                model_metadata,
                results_dir,
            )
            evaluation_wall_time_s = (
                time.perf_counter_ns() - evaluation_started_ns
            ) / 1_000_000_000
            status_after = request_status(session)
            validate_status(status_after, model_metadata)

        run_wall_time_s = (
            time.perf_counter_ns() - wall_started_ns
        ) / 1_000_000_000
        finished_at = datetime.now(timezone.utc).isoformat()
        summary = build_summary(
            rows,
            labels,
            predictions,
            model_metadata,
            status_before,
            status_after,
            started_at,
            finished_at,
            run_wall_time_s,
            evaluation_wall_time_s,
        )
        report_text = save_classification_results(
            labels,
            predictions,
            results_dir,
        )
        save_json(results_dir / "summary.json", summary)
        save_json(results_dir / "run.json", summary)
    except BaseException as error:
        run_metadata["status"] = "failed"
        run_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_metadata["error_type"] = type(error).__name__
        run_metadata["error"] = str(error)
        save_json(results_dir / "run.json", run_metadata)
        raise

    print(report_text)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Resultados: {results_dir}")


if __name__ == "__main__":
    main()
