import os
from preprocessing.ultrasound_preprocessor import UltrasoundPreprocessor
from configs.preprocessing_config import PreprocessingConfig


# ============================================================
# DATASET PATH
# ============================================================

DATASET_DIR = r"E:\PCOS PROJECT\Ovarian Ultra Sound image dataset\Ovarian_US"

# Output directory
PreprocessingConfig.OUTPUT_BASE_DIR = "preprocessing_outputs_dataset"


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# MAIN
# ============================================================

def main():

    preprocessor = UltrasoundPreprocessor()

    total_images = 0
    successful_images = 0
    failed_images = 0

    class_names = [
        "complex_cyst",
        "dominant_follicle",
        "healty",
        "poly_cyst",
        "simple_cyst"
    ]

    for class_name in class_names:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.isdir(class_dir):
            print(f"\nWARNING: Folder not found: {class_dir}")
            continue

        print("\n" + "=" * 60)
        print(f"Processing class: {class_name}")
        print("=" * 60)

        # Create class-specific output directories
        for stage_dir in PreprocessingConfig.STAGE_DIRECTORIES.values():
            os.makedirs(
                os.path.join(
                    PreprocessingConfig.OUTPUT_BASE_DIR,
                    class_name,
                    stage_dir
                ),
                exist_ok=True
            )

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        print(f"Images found: {len(image_files)}")

        for index, filename in enumerate(image_files, start=1):

            image_path = os.path.join(
                class_dir,
                filename
            )

            # Remove extension for image ID
            image_id = os.path.splitext(filename)[0]

            print(
                f"[{index}/{len(image_files)}] "
                f"{filename}"
            )

            try:

                final_image, stage_paths = (
                    preprocessor.preprocess_image(
                        image_path,
                        image_id=f"{class_name}_{image_id}"
                    )
                )

                if final_image is not None:

                    successful_images += 1

                    # Move/copy stage outputs into class folder
                    for stage, output_path in stage_paths.items():

                        class_stage_dir = os.path.join(
                            PreprocessingConfig.OUTPUT_BASE_DIR,
                            class_name,
                            PreprocessingConfig.STAGE_DIRECTORIES[stage]
                        )

                        os.makedirs(
                            class_stage_dir,
                            exist_ok=True
                        )

                        new_path = os.path.join(
                            class_stage_dir,
                            os.path.basename(output_path)
                        )

                        if os.path.exists(output_path):
                            os.replace(
                                output_path,
                                new_path
                            )

                else:
                    failed_images += 1

            except Exception as e:

                failed_images += 1

                print(
                    f"ERROR processing {filename}: {e}"
                )

            total_images += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Total images      : {total_images}")
    print(f"Successful        : {successful_images}")
    print(f"Failed            : {failed_images}")

    print(
        f"\nOutput directory: "
        f"{PreprocessingConfig.OUTPUT_BASE_DIR}"
    )


if __name__ == "__main__":
    main()