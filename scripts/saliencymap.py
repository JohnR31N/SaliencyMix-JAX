import argparse
import os

import cv2
import numpy as np
from torchvision import datasets
from tqdm import tqdm


def compute_saliency_map(img):
    """
    img: numpy array, shape [H, W, C], uint8, range 0-255
    return: saliency_map, shape [H, W], uint8
    """
    img_float = img.astype(np.float32) / 255.0

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(img_float)

    if not success:
        saliency_map = np.mean(img_float, axis=-1)

    saliency_map = (saliency_map * 255).astype(np.uint8)
    return saliency_map


def load_train_images(dataset, data_dir):
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        return train_dataset.data

    if dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
        return train_dataset.data

    if dataset == "svhn":
        train_dataset = datasets.SVHN(root=data_dir, split="train", download=True, transform=None)
        extra_dataset = datasets.SVHN(root=data_dir, split="extra", download=True, transform=None)
        images = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        images = np.transpose(images, (0, 2, 3, 1))  # [N,C,H,W] -> [N,H,W,C]
        return images

    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    images = load_train_images(args.dataset, args.data_dir)
    saliency_maps = []

    for img in tqdm(images):
        saliency_maps.append(compute_saliency_map(img))

    saliency_maps = np.stack(saliency_maps, axis=0)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_train_saliency.npy")
    np.save(output_path, saliency_maps)

    print("dataset:", args.dataset)
    print("saliency maps shape:", saliency_maps.shape)
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
