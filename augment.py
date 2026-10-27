# -- coding: utf-8 --
import shutil
from os import PathLike
from pathlib import Path
from typing import Union
import cv2
import numpy as np
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def augment_hue(
    input_folder: Union[str, PathLike],
    output_folder: Union[str, PathLike],
) -> None:
    """Augment every image in ``input_folder`` and save it in ``output_folder``."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise ValueError(f'Input folder does not exist: {input_folder}')

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for image_path in sorted(input_path.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in image_extensions
            or image_path.name.startswith('hue_')
        ):
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not read image: {image_path}')
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # print(f'{image_path.name} has mean values: L={np.mean(lab_image[:, :, 0]):.2f}, A={np.mean(lab_image[:, :, 1]):.2f}, B={np.mean(lab_image[:, :, 2]):.2f}')
        
        hue_factor = np.random.uniform(0.5, 1) * 255
        choice = np.random.uniform(0, 1)
        if choice < 0.5:
            # a channel (green) more, neutral is 128
            lab_image[:, :, 1] = np.clip(
                lab_image[:, :, 1] - hue_factor,
                0,
                255,
            )
        else:
            # b channel (blue) more, neutral is 128
            lab_image[:, :, 2] = np.clip(
                lab_image[:, :, 2] - hue_factor,
                0,
                255,
            )
        # print(f'After hue adjustment, {image_path.name} has mean values: L={np.mean(lab_image[:, :, 0]):.2f}, A={np.mean(lab_image[:, :, 1]):.2f}, B={np.mean(lab_image[:, :, 2]):.2f}')

        adjusted_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        destination = output_path / f'hue_{image_path.name}'
        if not cv2.imwrite(
            str(destination),
            adjusted_image
        ):
            raise ValueError(f'Could not write image: {destination}')

def copy_hue_ref(
    input_folder: Union[str, PathLike],
    output_folder: Union[str, PathLike],
    ) -> None:
    """Copy every image in ``input_folder`` to ``output_folder`` with a hue_ prefix."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise ValueError(f'Input folder does not exist: {input_folder}')

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for image_path in sorted(input_path.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in image_extensions
            or image_path.name.startswith('hue_')
        ):
            continue

        destination = output_path / f'hue_{image_path.name}'
        shutil.copy2(image_path, destination)

def copy_hue_txt(
    input_folder: Union[str, PathLike],
    output_folder: Union[str, PathLike],
    ) -> None:
    """Copy every text file in ``input_folder`` to ``output_folder`` with a hue_ prefix."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise ValueError(f'Input folder does not exist: {input_folder}')

    output_path.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(input_path.iterdir()):
        if (
            not txt_path.is_file()
            or txt_path.suffix.lower() != '.txt'
            or txt_path.name.startswith('hue_')
        ):
            continue

        destination = output_path / f'hue_{txt_path.name}'
        shutil.copy2(txt_path, destination)


def compute_metrics(result_dir: str, ref_dir: str) -> None:
    """Compute PSNR and SSIM metrics for images in ``result_dir`` against ``ref_dir``."""
    psnr_values = []
    ssim_values = []

    result_files = sorted([
        f for f in os.listdir(result_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])


    for filename in result_files:

        result_path = os.path.join(result_dir, filename)
        ref_path = os.path.join(ref_dir, filename)

        if not os.path.exists(ref_path):
            print("without Reference:", filename)
            continue

        result_img = Image.open(result_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        if ref_img.size != result_img.size:
            ref_img = ref_img.resize(result_img.size)

        result = np.asarray(result_img).astype(np.float32) / 255.0
        reference = np.asarray(ref_img).astype(np.float32) / 255.0

        psnr = peak_signal_noise_ratio(
            reference,
            result,
            data_range=1.0
        )

        ssim = structural_similarity(
            reference,
            result,
            channel_axis=-1,
            data_range=1.0
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    print(f"Average PSNR: {np.mean(psnr_values):.4f} dB")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")

if __name__ == '__main__':
    train_path = 'hue_data/train/raw/'
    ref_path = 'hue_data/train/ref/'
    txt_path = 'hue_data/train/txt/'
    
    augment_hue(train_path, train_path)
    copy_hue_ref(ref_path, ref_path)
    copy_hue_txt(txt_path, txt_path)
