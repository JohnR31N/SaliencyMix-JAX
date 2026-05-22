import argparse
import os

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state
from flax import struct

import orbax.checkpoint as ocp

from torchvision import datasets

from model.resnet import ResNet18


# -------------------------
# TrainState with BatchNorm
# -------------------------

class TrainState(train_state.TrainState):
    batch_stats: dict = struct.field(pytree_node=True)


# -------------------------
# Data utils
# -------------------------

CIFAR10_MEAN = np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255.0
CIFAR10_STD = np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255.0


def normalize_images(images):
    images = images.astype(np.float32) / 255.0
    images = (images - CIFAR10_MEAN) / CIFAR10_STD
    return images


def load_cifar10_test(data_dir="data"):
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None,
    )

    test_images = test_dataset.data
    test_labels = np.array(test_dataset.targets, dtype=np.int32)

    test_images = normalize_images(test_images)

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


# -------------------------
# Create empty state
# -------------------------

def create_empty_state(key, learning_rate=0.1):
    model = ResNet18(num_classes=10)

    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

    variables = model.init(
        key,
        dummy_input,
        train=True,
    )

    tx = optax.sgd(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )

    return state


# -------------------------
# Eval step
# -------------------------

@jax.jit
def eval_step(state, batch):
    images = batch["image"]
    labels = batch["label"]

    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }

    logits = state.apply_fn(
        variables,
        images,
        train=False,
        mutable=False,
    )

    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)

    return correct, labels.shape[0]


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/resnet18_saliencymix",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(args)
    print("JAX devices:", jax.devices())

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.checkpoint_dir}. "
            "Run train.py first."
        )

    key = jax.random.PRNGKey(args.seed)

    state = create_empty_state(key)

    checkpointer = ocp.PyTreeCheckpointer()

    state = checkpointer.restore(
        args.checkpoint_dir,
        item=state,
    )

    print(f"Loaded checkpoint from {args.checkpoint_dir}")

    test_images, test_labels = load_cifar10_test(args.data_dir)

    total_correct = 0
    total_num = 0

    test_total = (len(test_images) + args.batch_size - 1) // args.batch_size

    test_iter = batch_iterator(
        test_images,
        test_labels,
        batch_size=args.batch_size,
    )

    for batch in tqdm(test_iter, total=test_total):
        correct, num = eval_step(state, batch)

        total_correct += int(correct)
        total_num += int(num)

    test_acc = total_correct / total_num * 100.0

    print(f"test_acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()