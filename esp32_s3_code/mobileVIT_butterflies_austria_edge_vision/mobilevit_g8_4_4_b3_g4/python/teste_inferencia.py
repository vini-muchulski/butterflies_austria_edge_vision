import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
INPUT_INFO_PATH = Path(os.environ.get("INPUT_INFO_PATH", BASE_DIR / "input_info.json"))
DATASET_ROOT = Path(os.environ.get(
    "DATASET_ROOT",
    BASE_DIR.parents[3] / "Dataset-butterflies-austria",
))
TEST_DATASET_PATH = DATASET_ROOT / "test"


#IMAGE_INDEX = 299
IMAGE_INDEX = 199
#IMAGE_INDEX = 19
#IMAGE_INDEX = 640
#IMAGE_INDEX = 480






OUTPUT_PLOT_PATH = Path(os.environ.get(
    "OUTPUT_PLOT_PATH", BASE_DIR / "results" / f"butterflies_austria_plot_{IMAGE_INDEX}.png"
))
ESP32_IP = os.environ.get("ESP32_IP", "192.168.3.22")

#ESP32_IP =  "10.152.53.53"

PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
REQUEST_TIMEOUT = 300
SHOW_PLOT = True
RESIZE_SIZE = 288
CLASS_NAMES = [
    "Aglais_Urticae",
    "Anthocharis_Cardamines",
    "Apatura_Iris",
    "Argynnis_Paphia",
    "Colias_Myrmidone",
    "Gonepteryx_Rhamni",
    "Inachis_Io",
    "Iphiclides_Podalirius",
    "Lycaena_Virgaureae",
    "Lycaenidae",
    "Maniola_Jurtina",
    "Melanargia_Galathea",
    "Nymphalis_Antiopa",
    "Papilio_Machaon",
    "Parnassius_Apollo",
    "Pieris_Rapae",
    "Polygonia_C-album",
    "Vanessa_Atalanta",
    "Vanessa_Cardui",
    "Zerynthia_Polyxena",
]


def load_input_info(path):
    with open(path) as file:
        info = json.load(file)
    if (info["shape"] != [1, 3, 256, 256] or
            info["layout"] != "NCHW" or
            info["dtype"] != "int8" or
            info["scale"] <= 0):
        raise ValueError("Metadados de entrada incompatíveis")
    return info


def load_image(path):
    with Image.open(path) as image:
        return image.copy()


def get_test_samples():
    class_names = sorted(path.name for path in TEST_DATASET_PATH.iterdir() if path.is_dir())
    if class_names != CLASS_NAMES:
        raise ValueError("Classes do dataset diferem das classes do modelo")
    samples = []
    for label, class_name in enumerate(class_names):
        class_path = TEST_DATASET_PATH / class_name
        image_paths = sorted(
            path for path in class_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".ppm", ".bmp",
                ".pgm", ".tif", ".tiff", ".webp",
            }
        )
        samples.extend((path, label) for path in image_paths)
    return samples


def get_test_sample(index):
    samples = get_test_samples()
    if not 0 <= index < len(samples):
        raise IndexError(f"IMAGE_INDEX deve estar entre 0 e {len(samples) - 1}: {index}")
    image_path, true_label = samples[index]
    return load_image(image_path), true_label, image_path


def preprocess_image(image, info):
    _, channels, height, width = info["shape"]
    source_width, source_height = image.size
    if source_width <= source_height:
        resized_width = RESIZE_SIZE
        resized_height = int(RESIZE_SIZE * source_height / source_width)
    else:
        resized_height = RESIZE_SIZE
        resized_width = int(RESIZE_SIZE * source_width / source_height)
    resized = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    left = int(round((resized_width - width) / 2.0))
    top = int(round((resized_height - height) / 2.0))
    cropped = resized.crop((left, top, left + width, top + height)).convert("RGB")
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    bgr = pixels[:, :, ::-1]
    chw = np.ascontiguousarray(bgr.transpose(2, 0, 1))
    quantized = np.round(chw / float(info["scale"]) + int(info["zero_point"]))
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    if quantized.shape != (channels, height, width):
        raise ValueError(f"Shape de entrada inválido: {quantized.shape}")
    return quantized


def fnv1a32(data):
    value = 2166136261
    for byte in memoryview(np.ascontiguousarray(data)).cast("B"):
        value = ((value ^ int(byte)) * 16777619) & 0xFFFFFFFF
    return value


def send_bin(quantized):
    response = requests.post(
        PREDICT_URL,
        data=quantized.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=(5, REQUEST_TIMEOUT),
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise RuntimeError(result.get("error_message", "Falha na inferência"))
    return result


def plot_result(image, result, save_path=OUTPUT_PLOT_PATH, show_plot=SHOW_PLOT):
    import matplotlib

    if not show_plot:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    predicted = int(result["predicted_class"])
    figure, axis = plt.subplots()
    axis.imshow(image.convert("RGB"))
    axis.set_title(
        f"Label real: {result['true_name']} ({result['true_class']})\n"
        f"Label predita: {CLASS_NAMES[predicted]} ({predicted})"
    )
    axis.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(figure)


def main():
    info = load_input_info(INPUT_INFO_PATH)
    image, true_label, image_path = get_test_sample(IMAGE_INDEX)
    quantized = preprocess_image(image, info)
    result = send_bin(quantized)
    expected_hash = f"{fnv1a32(quantized):08x}"
    if result["input_fnv1a"] != expected_hash:
        raise RuntimeError("Hashes de entrada do PC e ESP32 divergem")
    predicted = int(result["predicted_class"])
    if len(result["scores"]) != len(CLASS_NAMES) or not 0 <= predicted < len(CLASS_NAMES):
        raise RuntimeError("Saída inválida")
    result["predicted_name"] = CLASS_NAMES[predicted]
    result["image_index"] = IMAGE_INDEX
    result["input_image"] = str(image_path)
    result["true_class"] = true_label
    result["true_name"] = CLASS_NAMES[true_label]
    result["correct"] = predicted == true_label
    print(json.dumps(result, indent=2))
    print(f"Inferência: {result['inference_us'] / 1_000_000:.6f} s")
    plot_result(image, result)


if __name__ == "__main__":
    main()
