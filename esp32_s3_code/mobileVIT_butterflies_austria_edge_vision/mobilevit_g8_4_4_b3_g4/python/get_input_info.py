import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("ABSL_LOGLEVEL", "3")

stderr_fd = sys.stderr.fileno()
saved_stderr_fd = os.dup(stderr_fd)
try:
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
finally:
    os.dup2(saved_stderr_fd, stderr_fd)
    os.close(saved_stderr_fd)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "data" / "mobilevit_int8.tflite"
OUTPUT_PATH = BASE_DIR / "input_info.json"

it = Interpreter(model_path=str(MODEL_PATH))
it.allocate_tensors()
d = it.get_input_details()[0]
info = {
    "shape": [int(value) for value in d["shape"]],
    "layout": "NCHW",
    "scale": float(d["quantization"][0]),
    "zero_point": int(d["quantization"][1]),
    "dtype": d["dtype"].__name__,
}
print(info)
OUTPUT_PATH.write_text(json.dumps(info, indent=2) + "\n")
