import numpy as np
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
import torch
import torchvision.transforms as T
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_image(baseline_path, folder_path):
    base_image = Image.open(baseline_path).convert("RGB")
    folder_image_paths = sorted([f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))])
    folder_images = []
    for image_file in folder_image_paths:
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path).convert("RGB")
        if img.size != base_image.size:
            print(f"Resizing {image_file} to match baseline size {base_image.size}")
            img = img.resize(base_image.size, Image.LANCZOS)
        if img.mode != base_image.mode:
            print(f"Converting {image_file} to match baseline mode {base_image.mode}")
            img = img.convert(base_image.mode)
        folder_images.append(img)
    return base_image, folder_images

def image_diff(img1, img2):
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    diff = np.abs(img1_np.astype(np.int32) - img2_np.astype(np.int32)).mean()
    return diff

def metrics_score(img1, img2):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    
    transform = T.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)
    
    lpips_score = lpips(img1_tensor, img2_tensor).item()
    psnr_score = psnr(img1_tensor, img2_tensor).item()
    ssim_score = ssim(img1_tensor, img2_tensor).item()
    
    return lpips_score, psnr_score, ssim_score

def main():
    parser = argparse.ArgumentParser(description="Compare two images using LPIPS, PSNR, SSIM, and pixel diff.")
    parser.add_argument("baseline", type=str, help="Path to the baseline image")
    parser.add_argument("folder", type=str, help="Path to the folder of images to compare")
    args = parser.parse_args()

    base_img, folder_imgs = read_image(args.baseline, args.folder)

    diffs = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    for img in folder_imgs:
        print(f"Comparing:\n - {args.baseline}\n - {img}")

        diff = image_diff(base_img, img)
        lpips_score, psnr_score, ssim_score = metrics_score(base_img, img)
        diffs.append(diff)
        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)

    print("\n=== Image Comparison Metrics ===")
    print(f"Pixel-wise Mean Absolute Difference: {np.mean(diffs):.4f}")
    print(f"LPIPS (VGG):                        {np.mean(lpips_scores):.4f}")
    print(f"PSNR:                              {np.mean(psnr_scores):.2f} dB")
    print(f"SSIM:                              {np.mean(ssim_scores):.4f}")

if __name__ == "__main__":
    main()