import sys

from inferencia_pc import MODEL_PATH, create_interpreter, run_inference
from teste_inferencia import CLASS_NAMES, IMAGE_INDEX, get_test_sample


def main():
    indices = [int(value) for value in sys.argv[1:]] or [IMAGE_INDEX]
    interpreter = create_interpreter(MODEL_PATH)
    correct = 0
    for index in indices:
        image, true_label, image_path = get_test_sample(index)
        result, _, _ = run_inference(
            MODEL_PATH,
            image,
            interpreter=interpreter,
        )
        hit = result["predicted_class"] == true_label
        correct += int(hit)
        print(
            f"Índice={index} imagem={image_path} "
            f"real={CLASS_NAMES[true_label]} ({true_label}) "
            f"previsão={result['predicted_name']} ({result['predicted_class']}) "
            f"score={result['confidence']:.6f} "
            f"acerto={hit} "
            f"input_hash={result['input_fnv1a']} "
            f"output_hash={result['output_fnv1a']}"
        )
    print(f"Acertos nas amostras selecionadas: {correct}/{len(indices)}")


if __name__ == "__main__":
    main()
