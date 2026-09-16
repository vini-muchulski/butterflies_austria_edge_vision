import json
import os
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from teste_inferencia import (
    CLASS_NAMES,
    IMAGE_INDEX,
    INPUT_INFO_PATH,
    fnv1a32,
    get_test_sample,
    load_input_info,
    preprocess_image,
)

IMAGE_INDEX = 299

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    BASE_DIR.parent / "data" / "model_int8_avgpool.tflite",
))
RESULTS_DIR = BASE_DIR / "results_pc"
SHOW_PLOT = True


def create_interpreter(model_path=MODEL_PATH, preserve_all_tensors=False):
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=preserve_all_tensors,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if list(input_details["shape"]) != [1, 3, 224, 224] or input_details["dtype"] != np.int8:
        raise ValueError("Entrada esperada: int8 [1,3,224,224]")
    if list(output_details["shape"]) != [1, 20] or output_details["dtype"] != np.int8:
        raise ValueError("Saída esperada: int8 [1,20]")
    if not np.isfinite(output_details["quantization"][0]) or output_details["quantization"][0] <= 0:
        raise ValueError("Quantização de saída inválida")
    info = load_input_info(INPUT_INFO_PATH)
    input_scale, input_zero_point = input_details["quantization"]
    if (info["shape"] != list(input_details["shape"]) or
            info["dtype"] != input_details["dtype"].__name__ or
            not np.isclose(info["scale"], input_scale) or
            info["zero_point"] != input_zero_point):
        raise ValueError("input_info.json difere do modelo")
    return interpreter


def prepare_input(image):
    return preprocess_image(image, load_input_info(INPUT_INFO_PATH))


def run_inference(model_path, image, interpreter=None):
    if interpreter is None:
        interpreter = create_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    quantized = prepare_input(image)
    interpreter.set_tensor(input_details["index"], quantized[None])
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"])[0]
    scale, zero_point = output_details["quantization"]
    scores = (raw.astype(np.float32) - zero_point) * scale
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores inválidos")
    predicted = int(np.argmax(scores))
    result = {
        "predicted_class": predicted,
        "predicted_name": CLASS_NAMES[predicted],
        "confidence": float(scores[predicted]),
        "scores": scores.tolist(),
        "input_fnv1a": f"{fnv1a32(quantized):08x}",
        "output_fnv1a": f"{fnv1a32(raw):08x}",
    }
    return result, quantized, raw


def plot_result(image, result, save_path, show_plot=SHOW_PLOT):
    import matplotlib

    if not show_plot:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.imshow(image.convert("RGB"))
    axis.set_title(
        f"Label real: {result['true_name']} ({result['true_class']})\n"
        f"Label predita: {result['predicted_name']} ({result['predicted_class']})"
    )
    axis.axis("off")
    figure.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(figure)


def main():
    image, true_label, image_path = get_test_sample(IMAGE_INDEX)
    result, quantized, raw = run_inference(MODEL_PATH, image)
    result["image_index"] = IMAGE_INDEX
    result["input_image"] = str(image_path)
    result["true_class"] = true_label
    result["true_name"] = CLASS_NAMES[true_label]
    result["correct"] = result["predicted_class"] == true_label
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    np.save(RESULTS_DIR / f"{stem}_input_int8_pc.npy", quantized)
    np.save(RESULTS_DIR / f"{stem}_output_int8_pc.npy", raw)
    (RESULTS_DIR / f"{stem}_result_pc.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    plot_result(image, result, RESULTS_DIR / f"{stem}_plot_pc.png")


if __name__ == "__main__":
    main()
