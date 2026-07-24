import numpy as np
from pathlib import Path


def inspect_npy_file(file_path):
    """
    Load and display detailed information about a .npy file.
    """

    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"\n❌ Error loading {file_path}: {e}")
        return

    print("\n" + "=" * 70)
    print(f"File: {file_path}")
    print("=" * 70)

    print(f"Shape          : {data.shape}")
    print(f"Dimensions     : {data.ndim}")
    print(f"Data Type      : {data.dtype}")
    print(f"Size           : {data.nbytes / (1024 * 1024):.4f} MB")

    print("\nStatistics")
    print("-" * 70)
    print(f"Minimum Value  : {np.min(data):.6f}")
    print(f"Maximum Value  : {np.max(data):.6f}")
    print(f"Mean Value     : {np.mean(data):.6f}")
    print(f"Contains NaN   : {np.isnan(data).any()}")
    print(f"Contains Inf   : {np.isinf(data).any()}")
    print(f"Zero Values    : {np.sum(data == 0)}")

    print("\nPreview")
    print("-" * 70)

    if data.ndim == 3:
        print("First Frame (first 5 landmarks):")
        print(data[0, :5, :])

    elif data.ndim == 2:
        print("First Frame (first 20 values):")
        print(data[0, :20])

    elif data.ndim == 1:
        print("First 20 values:")
        print(data[:20])

    else:
        print("Preview not supported.")

    print("=" * 70)


def load_dataset(dataset_path="dataset"):
    dataset_path = Path(dataset_path)
    return sorted(dataset_path.rglob("*.npy"))


def dataset_summary(npy_files):

    shapes = {}
    corrupted = 0
    nan_files = 0
    inf_files = 0
    classes = set()

    for file in npy_files:

        classes.add(file.parent.name)

        try:

            data = np.load(file)

            shapes[data.shape] = shapes.get(data.shape, 0) + 1

            if np.isnan(data).any():
                nan_files += 1

            if np.isinf(data).any():
                inf_files += 1

        except Exception:
            corrupted += 1

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total Classes      : {len(classes)}")
    print(f"Class Names        : {sorted(classes)}")
    print(f"Total Files        : {len(npy_files)}")
    print(f"Corrupted Files    : {corrupted}")
    print(f"Files with NaN     : {nan_files}")
    print(f"Files with Inf     : {inf_files}")

    print("\nShape Distribution")
    print("-" * 70)

    for shape, count in sorted(shapes.items()):
        print(f"{shape} --> {count} file(s)")

    print("=" * 70)


def list_files(npy_files):

    print("\nAvailable Files")
    print("=" * 70)

    for i, file in enumerate(npy_files, start=1):
        print(f"{i:3}. {file}")

    print("=" * 70)


def inspect_one_file(npy_files):

    list_files(npy_files)

    try:
        choice = int(input("\nEnter file number: "))

        if 1 <= choice <= len(npy_files):
            inspect_npy_file(npy_files[choice - 1])
        else:
            print("❌ Invalid selection.")

    except ValueError:
        print("❌ Please enter a valid number.")


def inspect_all_files(npy_files):

    for file in npy_files:
        inspect_npy_file(file)


def validate_dataset(npy_files):

    print("\nValidating Dataset...")
    print("=" * 70)

    valid = True

    expected_shape = None

    for file in npy_files:

        try:

            data = np.load(file)

            if expected_shape is None:
                expected_shape = data.shape

            if data.shape != expected_shape:
                print(f"Shape mismatch: {file}")
                print(f"Found: {data.shape}")
                print(f"Expected: {expected_shape}")
                valid = False

            if np.isnan(data).any():
                print(f"NaN values found: {file}")
                valid = False

            if np.isinf(data).any():
                print(f"Infinite values found: {file}")
                valid = False

        except Exception as e:

            print(f"Cannot load {file}: {e}")
            valid = False

    print()

    if valid:
        print("✅ Dataset validation PASSED.")
    else:
        print("❌ Dataset validation FAILED.")

    print("=" * 70)


def main():

    dataset_path = "dataset"

    npy_files = load_dataset(dataset_path)

    if not npy_files:
        print("No .npy files found.")
        return

    while True:

        print("\n")
        print("=" * 70)
        print("           FSL DATASET INSPECTOR")
        print("=" * 70)
        print("1. Show Dataset Summary")
        print("2. List Dataset Files")
        print("3. Inspect One File")
        print("4. Inspect All Files")
        print("5. Validate Dataset")
        print("6. Exit")
        print("=" * 70)

        choice = input("Select an option: ")

        if choice == "1":
            dataset_summary(npy_files)

        elif choice == "2":
            list_files(npy_files)

        elif choice == "3":
            inspect_one_file(npy_files)

        elif choice == "4":
            inspect_all_files(npy_files)

        elif choice == "5":
            validate_dataset(npy_files)

        elif choice == "6":
            print("\nGoodbye!")
            break

        else:
            print("❌ Invalid option.")


if __name__ == "__main__":
    main()