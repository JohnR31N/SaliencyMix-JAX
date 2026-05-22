import os
import numpy as np
import cv2
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


def main():
    os.makedirs("data", exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=None,
    )

    saliency_maps = []

    for img, label in tqdm(train_dataset):
        img = np.array(img)  # [H, W, C]
        saliency_map = compute_saliency_map(img)
        saliency_maps.append(saliency_map)

    saliency_maps = np.stack(saliency_maps, axis=0)

    print("saliency maps shape:", saliency_maps.shape)

    np.save("data/cifar10_train_saliency.npy", saliency_maps)


if __name__ == "__main__":
    main()