from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.utils.matlab_functions import imresize


FLICKR2K_URL = (
    "https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/"
    "Flickr2K.zip?download=true"
)
NTIRE_VALID_LR_ID = "1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj"
NTIRE_VALID_HR_ID = "1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_count(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir())


def download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl", "--location", "--fail", "--continue-at", "-",
        "--retry", "20", "--retry-all-errors", "--retry-delay", "5",
        "--output", str(destination), url,
    ]
    for attempt in range(1, 21):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        print(f"[retry] Download attempt {attempt}/20 failed; resuming in 5 seconds")
        time.sleep(5)
    raise RuntimeError(f"Download failed after 20 resumable attempts: {url}")


def locate_image_folder(root: Path, minimum: int, required_terms=()) -> Path:
    candidates = [root]
    candidates.extend(p for p in root.rglob("*") if p.is_dir())
    if required_terms:
        candidates = [
            path for path in candidates
            if all(term.lower() in str(path).lower() for term in required_terms)
        ]
    ranked = sorted(((image_count(p), p) for p in candidates), key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < minimum:
        raise RuntimeError(f"Could not find {minimum} extracted images below {root}")
    return ranked[0][1]


def prepare_flickr_hr(dataset_root: Path) -> Path:
    flickr_root = dataset_root / "Flickr2K"
    hr_dir = flickr_root / "Flickr2K_HR"
    if image_count(hr_dir) >= 2650:
        print(f"[skip] Flickr2K HR already exists: {hr_dir}")
        return hr_dir

    archive = dataset_root / "downloads" / "Flickr2K.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        print(f"[download] Flickr2K HR -> {archive}")
        download_with_resume(FLICKR2K_URL, archive)
    else:
        print(f"[reuse] Existing Flickr2K archive: {archive}")

    extract_root = flickr_root / "_extract_tmp"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(archive) as package:
        package.extractall(extract_root)
    try:
        source = locate_image_folder(extract_root, 2650, ("hr",))
    except RuntimeError:
        source = locate_image_folder(extract_root, 2650)
    try:
        bundled_lr = locate_image_folder(extract_root, 2650, ("lr", "x4"))
    except RuntimeError:
        bundled_lr = None
    hr_dir.mkdir(parents=True, exist_ok=True)
    for image in source.iterdir():
        if image.suffix.lower() in IMAGE_SUFFIXES:
            destination = hr_dir / image.name
            if not destination.exists():
                shutil.move(str(image), destination)
    if bundled_lr is not None:
        lr_dir = flickr_root / "Flickr2K_LR_bicubic" / "X4"
        lr_dir.mkdir(parents=True, exist_ok=True)
        for image in bundled_lr.iterdir():
            if image.suffix.lower() in IMAGE_SUFFIXES:
                destination = lr_dir / image.name
                if not destination.exists():
                    shutil.move(str(image), destination)
        print(f"[reuse] Extracted bundled Flickr2K x4 LR images: {lr_dir}")
    shutil.rmtree(extract_root)
    if image_count(hr_dir) < 2650:
        raise RuntimeError(f"Incomplete Flickr2K extraction: {hr_dir}")
    return hr_dir


def make_lr_one(arguments: tuple[Path, Path, int]) -> str:
    hr_path, lr_path, scale = arguments
    if lr_path.exists():
        return "skip"
    with Image.open(hr_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height = rgb.shape[0] - rgb.shape[0] % scale
    width = rgb.shape[1] - rgb.shape[1] % scale
    rgb = rgb[:height, :width]
    lr = imresize(rgb, scale=1.0 / scale, antialiasing=True)
    output = np.clip(np.round(lr * 255.0), 0, 255).astype(np.uint8)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(lr_path, compress_level=1)
    return "write"


def prepare_flickr_lr(hr_dir: Path, scale: int, workers: int) -> Path:
    lr_dir = hr_dir.parent / "Flickr2K_LR_bicubic" / f"X{scale}"
    hr_paths = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    tasks = [(path, lr_dir / f"{path.stem}x{scale}.png", scale) for path in hr_paths]
    missing = sum(not target.exists() for _, target, _ in tasks)
    if missing == 0 and len(tasks) >= 2650:
        print(f"[skip] Flickr2K x{scale} LR already exists: {lr_dir}")
        return lr_dir
    print(f"[generate] {missing} Flickr2K x{scale} LR images with MATLAB-style bicubic")
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, _ in enumerate(pool.map(make_lr_one, tasks), start=1):
            if index % 100 == 0 or index == len(tasks):
                print(f"  processed {index}/{len(tasks)}")
    return lr_dir


def check_div2k(dataset_root: Path, scale: int) -> tuple[Path, Path]:
    hr_dir = dataset_root / "DIV2K" / "HR"
    lr_dir = dataset_root / "DIV2K_bicubic" / "LR" / f"X{scale}"
    hr_count, lr_count = image_count(hr_dir), image_count(lr_dir)
    if hr_count < 900 or lr_count < 900:
        raise RuntimeError(
            f"DIV2K is incomplete: HR={hr_count}, bicubic LR x{scale}={lr_count}. "
            "This script will not overwrite or redownload a partial local copy."
        )
    print(f"[skip] DIV2K already exists: HR={hr_count}, bicubic LR x{scale}={lr_count}")
    return hr_dir, lr_dir


def ensure_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"Refusing to replace existing symlink: {destination}")
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    os.symlink(source.resolve(), destination)


def prepare_df2k_view(dataset_root: Path, scale: int) -> Path:
    """Create a zero-copy BasicSR view of DIV2K train + Flickr2K."""
    div_hr, div_lr = check_div2k(dataset_root, scale)
    flickr_hr = dataset_root / "Flickr2K" / "Flickr2K_HR"
    flickr_lr = dataset_root / "Flickr2K" / "Flickr2K_LR_bicubic" / f"X{scale}"
    if image_count(flickr_hr) < 2650 or image_count(flickr_lr) < 2650:
        raise RuntimeError("Flickr2K HR/LR is incomplete; run with --download-flickr2k first")

    df2k_root = dataset_root / "DF2K"
    target_hr = df2k_root / "HR"
    target_lr = df2k_root / "LR" / f"X{scale}"
    entries = []

    for index in range(1, 801):
        stem = f"{index:04d}"
        hr_source = div_hr / f"{stem}.png"
        lr_source = div_lr / f"{stem}x{scale}.png"
        target_stem = f"DIV2K_{stem}"
        ensure_symlink(hr_source, target_hr / f"{target_stem}.png")
        ensure_symlink(lr_source, target_lr / f"{target_stem}x{scale}.png")
        entries.append(f"{target_stem}.png (0,0,3)")

    flickr_images = sorted(
        path for path in flickr_hr.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for hr_source in flickr_images:
        lr_source = flickr_lr / f"{hr_source.stem}x{scale}.png"
        if not lr_source.is_file():
            raise RuntimeError(f"Missing Flickr2K pair: {lr_source}")
        target_stem = f"Flickr2K_{hr_source.stem}"
        ensure_symlink(hr_source, target_hr / f"{target_stem}.png")
        ensure_symlink(lr_source, target_lr / f"{target_stem}x{scale}.png")
        entries.append(f"{target_stem}.png (0,0,3)")

    meta_path = df2k_root / "meta_info_DF2K.txt"
    meta_path.write_text("\n".join(entries) + "\n", encoding="utf-8")

    div_val_meta = dataset_root / "DIV2K" / "meta_info_DIV2K_valid.txt"
    div_val_meta.write_text(
        "\n".join(f"{index:04d}.png (0,0,3)" for index in range(801, 901)) + "\n",
        encoding="utf-8")
    print(f"[ready] BasicSR DF2K view: {len(entries)} pairs -> {df2k_root}")
    return df2k_root


def gdown_file(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "gdown", file_id, "-O", str(destination)], check=True
    )


def extract_archive(archive: Path, output_root: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            package.extractall(output_root)
        return
    shutil.unpack_archive(str(archive), str(output_root))


def prepare_ntire_valid(dataset_root: Path) -> None:
    target = dataset_root / "NTIRE2026_ESR"
    lr_dir = target / "DIV2K_LSDIR_valid_LR"
    hr_dir = target / "DIV2K_LSDIR_valid_HR"
    if image_count(lr_dir) >= 200 and image_count(hr_dir) >= 200:
        print(f"[skip] NTIRE 2026 validation set already exists: {target}")
        return
    downloads = dataset_root / "downloads"
    packages = [
        (NTIRE_VALID_LR_ID, downloads / "DIV2K_LSDIR_valid_LR.zip"),
        (NTIRE_VALID_HR_ID, downloads / "DIV2K_LSDIR_valid_HR.zip"),
    ]
    for file_id, archive in packages:
        if not archive.exists():
            print(f"[download] Google Drive file {file_id} -> {archive}")
            gdown_file(file_id, archive)
        else:
            print(f"[reuse] Existing validation archive: {archive}")
        extract_archive(archive, target)
    print(f"NTIRE validation counts: LR={image_count(lr_dir)}, HR={image_count(hr_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SPANV2 FD2K datasets idempotently")
    parser.add_argument("--root", default="/home/qhm/datasets")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-flickr2k", action="store_true")
    parser.add_argument("--download-ntire-valid", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    check_div2k(dataset_root, args.scale)
    if args.download_flickr2k:
        hr_dir = prepare_flickr_hr(dataset_root)
        prepare_flickr_lr(hr_dir, args.scale, args.workers)
    flickr_hr = dataset_root / "Flickr2K" / "Flickr2K_HR"
    flickr_lr = dataset_root / "Flickr2K" / "Flickr2K_LR_bicubic" / f"X{args.scale}"
    if image_count(flickr_hr) >= 2650 and image_count(flickr_lr) >= 2650:
        prepare_df2k_view(dataset_root, args.scale)
    if args.download_ntire_valid:
        prepare_ntire_valid(dataset_root)


if __name__ == "__main__":
    main()
