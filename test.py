import argparse
import os
from functools import partial

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state
from flax import struct

import orbax.checkpoint as ocp

from torchvision import datasets

from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from model.wide_resnet import WideResNet28_10, WideResNet16_8


class TrainState(train_state.TrainState):
    batch_stats: dict = struct.field(pytree_node=True)


DATASET_STATS = {
    "cifar10": {
        "mean": np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255.0,
        "std": np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255.0,
        "num_classes": 10,
    },
    "cifar100": {
        "mean": np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255.0,
        "std": np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255.0,
        "num_classes": 100,
    },
    "svhn": {
        "mean": np.array([109.9, 109.7, 113.8], dtype=np.float32) / 255.0,
        "std": np.array([50.1, 50.6, 50.8], dtype=np.float32) / 255.0,
        "num_classes": 10,
    },
}


def get_dataset_config(dataset):
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_STATS[dataset]


def normalize_images(images, dataset):
    cfg = get_dataset_config(dataset)
    images = images.astype(np.float32) / 255.0
    images = (images - cfg["mean"]) / cfg["std"]
    return images


def load_test_dataset(dataset, data_dir="data"):
    if dataset == "cifar10":
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        test_images = test_dataset.data
        test_labels = np.array(test_dataset.targets, dtype=np.int32)

    elif dataset == "cifar100":
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
        test_images = test_dataset.data
        test_labels = np.array(test_dataset.targets, dtype=np.int32)

    elif dataset == "svhn":
        test_dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=None)
        test_images = np.transpose(test_dataset.data, (0, 2, 3, 1))
        test_labels = np.array(test_dataset.labels, dtype=np.int32)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_images = normalize_images(test_images, dataset)

    print("dataset:", dataset)
    print("test_images:", test_images.shape)
    print("test_labels:", test_labels.shape)

    return test_images, test_labels


def batch_iterator(images, labels, batch_size=128):
    n = images.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield {
            "image": jnp.array(images[start:end]),
            "label": jnp.array(labels[start:end]),
        }


def create_model(model_name, dataset, num_classes):
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes)
    if model_name == "resnet34":
        return ResNet34(num_classes=num_classes)
    if model_name == "resnet50":
        return ResNet50(num_classes=num_classes)
    if model_name == "resnet101":
        return ResNet101(num_classes=num_classes)
    if model_name == "wideresnet":
        if dataset == "svhn":
            return WideResNet16_8(num_classes=num_classes, drop_rate=0.4)
        return WideResNet28_10(num_classes=num_classes, drop_rate=0.3)
    raise ValueError(f"Unknown model: {model_name}")


def create_empty_state(key, dataset, model_name, num_classes):
    model = create_model(model_name, dataset, num_classes)
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

    variables = model.init({"params": key, "dropout": key}, dummy_input, train=True)

    tx = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


@partial(jax.jit, static_argnames=("num_classes",))
def eval_step(state, batch, num_classes):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == batch["label"])
    return correct, batch["label"].shape[0]


def default_checkpoint_dir(dataset, model, seed):
    return f"checkpoints/{dataset}_{model}_saliencymix_seed{seed}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "wideresnet"])

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]

    if args.checkpoint_dir is None:
        args.checkpoint_dir = default_checkpoint_dir(args.dataset, args.model, args.seed)

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    print(args)
    print("JAX devices:", jax.devices())

    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.checkpoint_dir}. Run train.py first."
        )

    key = jax.random.PRNGKey(args.seed)
    state = create_empty_state(key, args.dataset, args.model, num_classes)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(args.checkpoint_dir, item=state)

    print(f"Loaded checkpoint from {args.checkpoint_dir}")

    test_images, test_labels = load_test_dataset(args.dataset, args.data_dir)

    total_correct = 0
    total_num = 0
    test_total = (len(test_images) + args.batch_size - 1) // args.batch_size

    for batch in tqdm(batch_iterator(test_images, test_labels, args.batch_size), total=test_total):
        correct, num = eval_step(state, batch, num_classes)
        total_correct += int(correct)
        total_num += int(num)

    test_acc = total_correct / total_num * 100.0
    test_error = 100.0 - test_acc

    print(f"test_acc: {test_acc:.2f}%")
    print(f"test_error: {test_error:.2f}%")


if __name__ == "__main__":
    main()
