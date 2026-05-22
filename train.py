import argparse
import csv
import os

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state
from flax import struct

from torchvision import datasets

from model.resnet import ResNet18
from augmentations.saliencymix import saliencymix_batch

import orbax.checkpoint as ocp


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
    """
    images: uint8 [N, H, W, C], range 0-255
    return: float32 normalized [N, H, W, C]
    """
    images = images.astype(np.float32) / 255.0
    images = (images - CIFAR10_MEAN) / CIFAR10_STD
    return images


def load_cifar10_with_saliency(data_dir="data"):
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None,
    )

    train_images = train_dataset.data
    train_labels = np.array(train_dataset.targets, dtype=np.int32)

    test_images = test_dataset.data
    test_labels = np.array(test_dataset.targets, dtype=np.int32)

    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    saliency_path = os.path.join(data_dir, "cifar10_train_saliency.npy")

    if not os.path.exists(saliency_path):
        raise FileNotFoundError(
            f"Cannot find {saliency_path}. "
            "Run: python scripts/saliencymap.py first."
        )

    train_saliency = np.load(saliency_path).astype(np.float32)

    print("train_images:", train_images.shape)
    print("train_labels:", train_labels.shape)
    print("train_saliency:", train_saliency.shape)
    print("test_images:", test_images.shape)
    print("test_labels:", test_labels.shape)

    return train_images, train_labels, train_saliency, test_images, test_labels


def batch_iterator(images, labels, saliency_maps=None, batch_size=128, shuffle=True):
    n = images.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]

        batch = {
            "image": jnp.array(images[batch_idx]),
            "label": jnp.array(labels[batch_idx]),
        }

        if saliency_maps is not None:
            batch["saliency"] = jnp.array(saliency_maps[batch_idx])

        yield batch


def random_crop_flip_batch(key, images, saliency_maps, apply_aug=True):
    """
    Reproduce CIFAR-style data augmentation:
      - RandomCrop(32, padding=4)
      - RandomHorizontalFlip(p=0.5)

    Important:
    The same crop/flip is applied to both image and saliency map,
    otherwise the saliency map will no longer align with the image.

    images: [B, 32, 32, 3], already normalized
    saliency_maps: [B, 32, 32]
    """
    B, H, W, C = images.shape
    padding = 4

    key_y, key_x, key_flip = jax.random.split(key, 3)

    # For padding=4 and crop size=32, offsets are integers in [0, 8].
    crop_y = jax.random.randint(
        key_y,
        shape=(B,),
        minval=0,
        maxval=2 * padding + 1,
    )

    crop_x = jax.random.randint(
        key_x,
        shape=(B,),
        minval=0,
        maxval=2 * padding + 1,
    )

    flip = jax.random.bernoulli(
        key_flip,
        p=0.5,
        shape=(B,),
    )

    # Original PyTorch transform pads raw pixels with 0 before normalization.
    # Our images are already normalized, so raw zero corresponds to:
    # normalized_zero = (0 - mean) / std
    pad_value = jnp.array(
        -CIFAR10_MEAN / CIFAR10_STD,
        dtype=jnp.float32,
    )

    padded_images = jnp.ones(
        (B, H + 2 * padding, W + 2 * padding, C),
        dtype=images.dtype,
    ) * pad_value.reshape(1, 1, 1, C)

    padded_images = padded_images.at[
        :, padding:padding + H, padding:padding + W, :
    ].set(images)

    padded_saliency = jnp.pad(
        saliency_maps,
        pad_width=((0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    def crop_flip_one(img, sal, y, x, do_flip):
        cropped_img = jax.lax.dynamic_slice(
            img,
            start_indices=(y, x, 0),
            slice_sizes=(H, W, C),
        )

        cropped_sal = jax.lax.dynamic_slice(
            sal,
            start_indices=(y, x),
            slice_sizes=(H, W),
        )

        flipped_img = jnp.flip(cropped_img, axis=1)
        flipped_sal = jnp.flip(cropped_sal, axis=1)

        cropped_img = jnp.where(do_flip, flipped_img, cropped_img)
        cropped_sal = jnp.where(do_flip, flipped_sal, cropped_sal)

        return cropped_img, cropped_sal

    aug_images, aug_saliency = jax.vmap(crop_flip_one)(
        padded_images,
        padded_saliency,
        crop_y,
        crop_x,
        flip,
    )

    images = jnp.where(apply_aug, aug_images, images)
    saliency_maps = jnp.where(apply_aug, aug_saliency, saliency_maps)

    return images, saliency_maps


# -------------------------
# Loss / metrics
# -------------------------

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return jnp.mean(loss)


def accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


# -------------------------
# Train / eval step
# -------------------------

@jax.jit
def train_step(state, batch, key, beta, salmix_prob, data_augmentation):
    def loss_fn(params):
        images = batch["image"]          # [B, 32, 32, 3]
        labels = batch["label"]          # [B]
        saliency = batch["saliency"]     # [B, 32, 32]

        key_aug, key_mix = jax.random.split(key)

        images, saliency = random_crop_flip_batch(
            key_aug,
            images,
            saliency,
            apply_aug=data_augmentation,
        )

        mixed_images, labels_a, labels_b, lam = saliencymix_batch(
            key=key_mix,
            images=images,
            labels=labels,
            saliency_maps=saliency,
            beta=beta,
            salmix_prob=salmix_prob,
        )

        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        logits, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )

        loss_a = cross_entropy_loss(logits, labels_a)
        loss_b = cross_entropy_loss(logits, labels_b)

        loss = lam * loss_a + (1.0 - lam) * loss_b

        # This is only a rough training metric because mixed labels are soft.
        acc = accuracy(logits, labels_a)

        metrics = {
            "loss": loss,
            "acc": acc,
            "lam": lam,
        }

        return loss, (metrics, new_model_state)

    (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)

    state = state.replace(
        batch_stats=new_model_state["batch_stats"]
    )

    return state, metrics


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

    loss = cross_entropy_loss(logits, labels)
    acc = accuracy(logits, labels)

    return {
        "loss": loss,
        "acc": acc,
    }


# -------------------------
# Create state
# -------------------------

def create_train_state(
    key,
    learning_rate,
    momentum,
    weight_decay,
    steps_per_epoch,
):
    model = ResNet18(num_classes=10)

    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

    variables = model.init(
        key,
        dummy_input,
        train=True,
    )

    params = variables["params"]
    batch_stats = variables["batch_stats"]

    # Original CIFAR schedule:
    # epoch 60, 120, 160, gamma=0.2
    schedule = optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales={
            60 * steps_per_epoch: 0.2,
            120 * steps_per_epoch: 0.2,
            160 * steps_per_epoch: 0.2,
        },
    )

    tx = optax.sgd(
        learning_rate=schedule,
        momentum=momentum,
        nesterov=True,
    )

    if weight_decay > 0:
        tx = optax.chain(
            optax.add_decayed_weights(weight_decay),
            tx,
        )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--salmix_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        default=False,
        help="Use RandomCrop(32, padding=4) and RandomHorizontalFlip.",
    )
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/resnet18_saliencymix",
    )

    args = parser.parse_args()

    print(args)
    print("JAX devices:", jax.devices())

    train_images, train_labels, train_saliency, test_images, test_labels = (
        load_cifar10_with_saliency(args.data_dir)
    )

    # Make NumPy-side batch shuffling deterministic for each seed.
    np.random.seed(args.seed)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    steps_per_epoch = (len(train_images) + args.batch_size - 1) // args.batch_size

    state = create_train_state(
        key=init_key,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        steps_per_epoch=steps_per_epoch,
    )

    best_test_acc = 0.0

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args.log_dir = os.path.abspath(args.log_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()

    csv_path = os.path.join(
        args.log_dir,
        f"cifar10_resnet18_seed{args.seed}.csv",
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "best_test_acc",
            "mean_lam",
            "seed",
            "data_augmentation",
            "learning_rate",
            "batch_size",
            "beta",
            "salmix_prob",
        ],
    )
    csv_writer.writeheader()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"logging to: {csv_path}")

    train_total = (len(train_images) + args.batch_size - 1) // args.batch_size
    test_total = (len(test_images) + args.batch_size - 1) // args.batch_size

    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_losses = []
            train_accs = []
            train_lams = []

            train_iter = batch_iterator(
                train_images,
                train_labels,
                train_saliency,
                batch_size=args.batch_size,
                shuffle=True,
            )

            progress = tqdm(train_iter, total=train_total)

            for batch in progress:
                key, step_key = jax.random.split(key)

                state, metrics = train_step(
                    state=state,
                    batch=batch,
                    key=step_key,
                    beta=args.beta,
                    salmix_prob=args.salmix_prob,
                    data_augmentation=args.data_augmentation,
                )

                loss = float(metrics["loss"])
                acc = float(metrics["acc"])
                lam = float(metrics["lam"])

                train_losses.append(loss)
                train_accs.append(acc)
                train_lams.append(lam)

                progress.set_postfix(
                    loss=f"{np.mean(train_losses):.4f}",
                    acc=f"{np.mean(train_accs) * 100:.2f}",
                    lam=f"{np.mean(train_lams):.3f}",
                )

            # Evaluation
            test_losses = []
            test_accs = []

            test_iter = batch_iterator(
                test_images,
                test_labels,
                saliency_maps=None,
                batch_size=args.batch_size,
                shuffle=False,
            )

            for batch in tqdm(test_iter, total=test_total):
                metrics = eval_step(
                    state=state,
                    batch=batch,
                )

                test_losses.append(float(metrics["loss"]))
                test_accs.append(float(metrics["acc"]))

            train_loss = float(np.mean(train_losses))
            train_acc = float(np.mean(train_accs) * 100.0)
            mean_lam = float(np.mean(train_lams))
            test_loss = float(np.mean(test_losses))
            test_acc = float(np.mean(test_accs) * 100.0)

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.2f}, "
                f"test_loss={test_loss:.4f}, "
                f"test_acc={test_acc:.2f}"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc

                checkpointer.save(
                    args.checkpoint_dir,
                    state,
                    force=True,
                )

                print(f"saved best checkpoint to {args.checkpoint_dir}")

            csv_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_test_acc": float(best_test_acc),
                    "mean_lam": mean_lam,
                    "seed": args.seed,
                    "data_augmentation": args.data_augmentation,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "beta": args.beta,
                    "salmix_prob": args.salmix_prob,
                }
            )
            csv_file.flush()

            print(f"best_test_acc={best_test_acc:.2f}")

    finally:
        csv_file.close()

    print("Training finished.")


if __name__ == "__main__":
    main()
