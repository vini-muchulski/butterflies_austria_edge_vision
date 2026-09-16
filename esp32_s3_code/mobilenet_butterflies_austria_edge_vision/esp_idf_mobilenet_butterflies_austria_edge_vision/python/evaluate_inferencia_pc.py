import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import seaborn as sns
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from inferencia_pc import MODEL_PATH, run_inference
from teste_inferencia import (
    CLASS_NAMES,
    TEST_DATASET_PATH,
    get_test_samples,
    load_image,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_pc" / "evaluation"


def create_interpreter(resolver_type):
    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        experimental_op_resolver_type=resolver_type,
    )
    interpreter.allocate_tensors()
    return interpreter


def save_predictions(rows, results_dir):
    fieldnames = [
        "image_index",
        "image_path",
        "true_class",
        "true_name",
        "predicted_class",
        "predicted_name",
        "correct",
        "predicted_logit",
        "input_fnv1a",
        "output_fnv1a",
        "scores",
    ]
    with (results_dir / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(matrix, results_dir):
    with (results_dir / "confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["real/predita", *CLASS_NAMES])
        for class_name, values in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *values.tolist()])

    import matplotlib.pyplot as plt

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
    axis.set_title("Matriz de confusão - Butterflies Austria MobileNetV2 INT8")
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png")
    plt.close(figure)


def evaluate(samples, resolver_name, resolver_type):
    results_dir = RESULTS_DIR / resolver_name
    interpreter = create_interpreter(resolver_type)
    labels = []
    predictions = []
    rows = []

    for image_index, (image_path, true_label) in enumerate(
        tqdm(samples, desc=f"Avaliando {resolver_name}")
    ):
        image = load_image(image_path)
        result, _, _ = run_inference(
            MODEL_PATH,
            image,
            interpreter=interpreter,
        )
        predicted_class = result["predicted_class"]
        labels.append(true_label)
        predictions.append(predicted_class)
        rows.append({
            "image_index": image_index,
            "image_path": str(image_path.relative_to(TEST_DATASET_PATH)),
            "true_class": true_label,
            "true_name": CLASS_NAMES[true_label],
            "predicted_class": predicted_class,
            "predicted_name": CLASS_NAMES[predicted_class],
            "correct": predicted_class == true_label,
            "predicted_logit": result["confidence"],
            "input_fnv1a": result["input_fnv1a"],
            "output_fnv1a": result["output_fnv1a"],
            "scores": json.dumps(result["scores"], separators=(",", ":")),
        })

    class_indices = list(range(len(CLASS_NAMES)))
    accuracy = accuracy_score(labels, predictions)
    report_text = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_data = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_indices)

    results_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(rows, results_dir)
    save_confusion_matrix(matrix, results_dir)
    (results_dir / "classification_report.txt").write_text(report_text + "\n")
    (results_dir / "classification_report.json").write_text(
        json.dumps(report_data, indent=2) + "\n"
    )

    summary = {
        "model_path": str(MODEL_PATH),
        "dataset_path": str(TEST_DATASET_PATH),
        "resolver": resolver_name,
        "samples": len(samples),
        "correct": int(np.sum(np.asarray(labels) == np.asarray(predictions))),
        "incorrect": int(np.sum(np.asarray(labels) != np.asarray(predictions))),
        "accuracy": float(accuracy),
        "classes": CLASS_NAMES,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"Resolver: {resolver_name}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Amostras: {len(samples)}")
    print(f"Acurácia: {accuracy:.4f}")
    print(report_text)
    print(f"Resultados: {results_dir}")


def main():
    samples = get_test_samples()
    resolvers = {
        "builtin_without_default_delegates": (
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
        "builtin_ref": OpResolverType.BUILTIN_REF,
    }
    for resolver_name, resolver_type in resolvers.items():
        evaluate(samples, resolver_name, resolver_type)


if __name__ == "__main__":
    main()
