from pathlib import Path
from shutil import copy2

from sklearn.model_selection import train_test_split


SEED = 42
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "butterflies"
SPLIT_DIRS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR / "val",
    "test": BASE_DIR / "test",
}


def collect_samples():
    class_directories = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())
    samples = []

    for class_directory in class_directories:
        for image_path in sorted(class_directory.rglob("*")):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((image_path, class_directory.name))

    if not samples:
        raise RuntimeError(f"Nenhuma imagem encontrada em {SOURCE_DIR}")

    return samples, [path.name for path in class_directories]


def split_samples(samples):
    labels = [class_name for _, class_name in samples]
    train_samples, remainder_samples = train_test_split(
        samples,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    remainder_labels = [class_name for _, class_name in remainder_samples]
    val_samples, test_samples = train_test_split(
        remainder_samples,
        test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
        random_state=SEED,
        stratify=remainder_labels,
    )
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def create_split_directories(class_names):
    existing_directories = [path for path in SPLIT_DIRS.values() if path.exists()]

    if existing_directories:
        paths = ", ".join(str(path) for path in existing_directories)
        raise FileExistsError(f"Diretorios de destino ja existem: {paths}")

    for split_directory in SPLIT_DIRS.values():
        for class_name in class_names:
            (split_directory / class_name).mkdir(parents=True)


def copy_samples(split_samples_map):
    for split_name, samples in split_samples_map.items():
        for source_path, class_name in samples:
            relative_path = source_path.relative_to(SOURCE_DIR / class_name)
            destination_path = SPLIT_DIRS[split_name] / class_name / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(source_path, destination_path)


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)

    samples, class_names = collect_samples()
    split_samples_map = split_samples(samples)
    create_split_directories(class_names)
    copy_samples(split_samples_map)

    print(f"Classes: {len(class_names)}")
    for split_name, split_items in split_samples_map.items():
        print(f"{split_name}: {len(split_items)}")


if __name__ == "__main__":
    main()
