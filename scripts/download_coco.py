"""
COCO Dataset Downloader
=======================

Utility script to download COCO dataset.
"""

import argparse
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm


def download_file(url: str, destination: str) -> None:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        destination: Local path to save file
    """
    print(f"Downloading: {url}")
    print(f"Saving to: {destination}")

    # Download with progress
    urllib.request.urlretrieve(url, destination)

    print(f"Download completed: {destination}")


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract ZIP file.

    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
    """
    print(f"Extracting: {zip_path}")
    print(f"Extracting to: {extract_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction completed")


def download_coco_split(
    split: str,
    output_dir: str,
    download_images: bool = True,
    download_annotations: bool = True
) -> None:
    """
    Download COCO dataset split.

    Args:
        split: Dataset split (train2017, val2017)
        output_dir: Output directory
        download_images: Whether to download images
        download_annotations: Whether to download annotations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # COCO URLs
    base_url = "http://images.cocodataset.org"

    # Download images
    if download_images:
        print(f"\n{'='*60}")
        print(f"Downloading COCO {split} images...")
        print(f"{'='*60}")

        images_url = f"{base_url}/zips/{split}.zip"
        images_zip = str(output_path / f"{split}.zip")
        images_dir = str(output_path / "images" / split)

        download_file(images_url, images_zip)
        extract_zip(images_zip, images_dir)

        # Remove zip file
        os.remove(images_zip)
        print(f"Removed: {images_zip}")

    # Download annotations
    if download_annotations:
        print(f"\n{'='*60}")
        print("Downloading COCO annotations...")
        print(f"{'='*60}")

        annotations_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        annotations_zip = str(output_path / "annotations.zip")
        annotations_dir = str(output_path / "annotations")

        download_file(annotations_url, annotations_zip)
        extract_zip(annotations_zip, annotations_dir)

        # Remove zip file
        os.remove(annotations_zip)
        print(f"Removed: {annotations_zip}")

    print(f"\n{'='*60}")
    print("Download completed!")
    print(f"{'='*60}")
    print(f"COCO dataset saved to: {output_path}")


def download_coco_vqa(
    output_dir: str
) -> None:
    """
    Download VQA annotations for COCO.

    Args:
        output_dir: Output directory
    """
    print(f"\n{'='*60}")
    print("Downloading VQA annotations...")
    print(f"{'='*60}")

    output_path = Path(output_dir) / "annotations"
    output_path.mkdir(parents=True, exist_ok=True)

    # VQA URLs (example - actual URLs may vary)
    vqa_urls = [
        "https://s3.amazonaws.com/cocodataset/vqa/v2_mscoco_val2014_questions.json",
        "https://s3.amazonaws.com/cocodataset/vqa/v2_mscoco_train2014_questions.json",
    ]

    for url in vqa_urls:
        filename = url.split('/')[-1]
        destination = str(output_path / filename)
        download_file(url, destination)

    print("VQA annotations downloaded")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download COCO dataset"
    )

    parser.add_argument(
        '--split',
        type=str,
        default='val2017',
        choices=['train2017', 'val2017', 'train2014', 'val2014'],
        help='Dataset split to download'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./data/coco',
        help='Output directory'
    )

    parser.add_argument(
        '--images',
        action='store_true',
        default=True,
        help='Download images'
    )

    parser.add_argument(
        '--annotations',
        action='store_true',
        default=True,
        help='Download annotations'
    )

    parser.add_argument(
        '--vqa',
        action='store_true',
        default=False,
        help='Download VQA annotations'
    )

    args = parser.parse_args()

    print("="*60)
    print("COCO Dataset Downloader")
    print("="*60)

    print(f"\nConfiguration:")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output}")
    print(f"  Images: {args.images}")
    print(f"  Annotations: {args.annotations}")
    print(f"  VQA: {args.vqa}")

    try:
        download_coco_split(
            split=args.split,
            output_dir=args.output,
            download_images=args.images,
            download_annotations=args.annotations
        )

        if args.vqa:
            download_coco_vqa(args.output)

        print("\n" + "="*60)
        print("All downloads completed successfully!")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        print("Download failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
