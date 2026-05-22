import argparse
import csv
import os
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state
from flax import struct

from torchvision import datasets

from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from model.wide_resnet import WideResNet28_10, WideResNet16_8
from augmentations.saliencymix import saliencymix_batch

import orbax.checkpoint as ocp


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


def load_dataset_with_saliency(dataset, data_dir="data"):
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        train_images = train_dataset.data
        train_labels = np.array(train_dataset.targets, dtype=np.int32)
        test_images = test_dataset.data
        test_labels = np.array(test_dataset.targets, dtype=np.int32)

    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
        train_images = train_dataset.data
        train_labels = np.array(train_dataset.targets, dtype=np.int32)
        test_images = test_dataset.data
        test_labels = np.array(test_dataset.targets, dtype=np.int32)

    elif dataset == "svhn":
        train_dataset = datasets.SVHN(root=data_dir, split="train", download=True, transform=None)
        extra_dataset = datasets.SVHN(root=data_dir, split="extra", download=True, transform=None)
        test_dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=None)

        train_images = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        train_images = np.transpose(train_images, (0, 2, 3, 1))
        train_labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0).astype(np.int32)

        test_images = np.transpose(test_dataset.data, (0, 2, 3, 1))
        test_labels = np.array(test_dataset.labels, dtype=np.int32)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_images = normalize_images(train_images, dataset)
    test_images = normalize_images(test_images, dataset)

    saliency_path = os.path.join(data_dir, f"{dataset}_train_saliency.npy")
    if not os.path.exists(saliency_path):
        raise FileNotFoundError(
            f"Cannot find {saliency_path}. Run: python scripts/saliencymap.py --dataset {dataset}"
        )

    train_saliency = np.load(saliency_path).astype(np.float32)
    if len(train_saliency) != len(train_images):
        raise ValueError(
            f"Saliency map count mismatch: got {len(train_saliency)}, expected {len(train_images)} for {dataset}."
        )

    print("dataset:", dataset)
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
        batch = {"image": jnp.array(images[batch_idx]), "label": jnp.array(labels[batch_idx])}
        if saliency_maps is not None:
            batch["saliency"] = jnp.array(saliency_maps[batch_idx])
        yield batch


def random_crop_flip_batch(key, images, saliency_maps, pad_value, apply_aug=True):
    B, H, W, C = images.shape
    padding = 4
    key_y, key_x, key_flip = jax.random.split(key, 3)

    crop_y = jax.random.randint(key_y, shape=(B,), minval=0, maxval=2 * padding + 1)
    crop_x = jax.random.randint(key_x, shape=(B,), minval=0, maxval=2 * padding + 1)
    flip = jax.random.bernoulli(key_flip, p=0.5, shape=(B,))

    padded_images = jnp.ones((B, H + 2 * padding, W + 2 * padding, C), dtype=images.dtype) * pad_value.reshape(1, 1, 1, C)
    padded_images = padded_images.at[:, padding:padding + H, padding:padding + W, :].set(images)

    padded_saliency = jnp.pad(
        saliency_maps,
        pad_width=((0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    def crop_flip_one(img, sal, y, x, do_flip):
        cropped_img = jax.lax.dynamic_slice(img, start_indices=(y, x, 0), slice_sizes=(H, W, C))
        cropped_sal = jax.lax.dynamic_slice(sal, start_indices=(y, x), slice_sizes=(H, W))
        flipped_img = jnp.flip(cropped_img, axis=1)
        flipped_sal = jnp.flip(cropped_sal, axis=1)
        cropped_img = jnp.where(do_flip, flipped_img, cropped_img)
        cropped_sal = jnp.where(do_flip, flipped_sal, cropped_sal)
        return cropped_img, cropped_sal

    aug_images, aug_saliency = jax.vmap(crop_flip_one)(padded_images, padded_saliency, crop_y, crop_x, flip)
    images = jnp.where(apply_aug, aug_images, images)
    saliency_maps = jnp.where(apply_aug, aug_saliency, saliency_maps)
    return images, saliency_maps


def cross_entropy_loss(logits, labels, num_classes):
    one_hot = jax.nn.one_hot(labels, num_classes)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


@partial(jax.jit, static_argnames=("num_classes",))
def train_step(state, batch, key, beta, salmix_prob, data_augmentation, pad_value, num_classes):
    def loss_fn(params):
        images = batch["image"]
        labels = batch["label"]
        saliency = batch["saliency"]

        key_aug, key_mix, key_dropout = jax.random.split(key, 3)
        images, saliency = random_crop_flip_batch(
            key_aug, images, saliency, pad_value=pad_value, apply_aug=data_augmentation
        )

        mixed_images, labels_a, labels_b, lam = saliencymix_batch(
            key=key_mix,
            images=images,
            labels=labels,
            saliency_maps=saliency,
            beta=beta,
            salmix_prob=salmix_prob,
        )

        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": key_dropout},
        )

        loss_a = cross_entropy_loss(logits, labels_a, num_classes)
        loss_b = cross_entropy_loss(logits, labels_b, num_classes)
        loss = lam * loss_a + (1.0 - lam) * loss_b

        metrics = {"loss": loss, "acc": accuracy(logits, labels_a), "lam": lam}
        return loss, (metrics, new_model_state)

    (loss, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    return state, metrics


@partial(jax.jit, static_argnames=("num_classes",))
def eval_step(state, batch, num_classes):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return {"loss": cross_entropy_loss(logits, batch["label"], num_classes), "acc": accuracy(logits, batch["label"])}


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


def create_lr_schedule(dataset, learning_rate, steps_per_epoch):
    if dataset == "svhn":
        boundaries_and_scales = {80 * steps_per_epoch: 0.1, 120 * steps_per_epoch: 0.1}
    else:
        boundaries_and_scales = {
            60 * steps_per_epoch: 0.2,
            120 * steps_per_epoch: 0.2,
            160 * steps_per_epoch: 0.2,
        }

    return optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales=boundaries_and_scales,
    )


def create_train_state(key, dataset, model_name, learning_rate, momentum, weight_decay, steps_per_epoch, num_classes):
    model = create_model(model_name, dataset, num_classes)
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

    variables = model.init({"params": key, "dropout": key}, dummy_input, train=True)
    tx = optax.sgd(learning_rate=create_lr_schedule(dataset, learning_rate, steps_per_epoch), momentum=momentum, nesterov=True)

    if weight_decay > 0:
        tx = optax.chain(optax.add_decayed_weights(weight_decay), tx)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


def save_training_plots(history, figure_dir, dataset, model_name, seed):
    os.makedirs(figure_dir, exist_ok=True)
    epochs = history["epoch"]
    prefix = f"{dataset}_{model_name}_seed{seed}"

    def save_plot(y_keys, labels, title, ylabel, filename):
        plt.figure()
        for y_key, label in zip(y_keys, labels):
            plt.plot(epochs, history[y_key], marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"{prefix}_{filename}"), dpi=200)
        plt.close()

    save_plot(["train_loss", "test_loss"], ["Train Loss", "Test Loss"], "Loss vs Epoch", "Loss", "loss_curve.png")
    save_plot(["train_acc", "test_acc"], ["Train Accuracy", "Test Accuracy"], "Accuracy vs Epoch", "Accuracy (%)", "accuracy_curve.png")
    save_plot(["test_error"], ["Test Error"], "Test Error vs Epoch", "Test Error (%)", "test_error_curve.png")
    save_plot(["best_test_acc"], ["Best Test Accuracy"], "Best Test Accuracy vs Epoch", "Accuracy (%)", "best_test_accuracy_curve.png")
    save_plot(["mean_lam"], ["Mean Lambda"], "Mean Lambda vs Epoch", "Lambda", "mean_lambda_curve.png")
    print(f"Saved plots to: {figure_dir}")


def default_checkpoint_dir(dataset, model, seed):
    return f"checkpoints/{dataset}_{model}_saliencymix_seed{seed}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "wideresnet"])

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--salmix_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_augmentation", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--figure_dir", type=str, default="figures")
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    pad_value = jnp.array(-cfg["mean"] / cfg["std"], dtype=jnp.float32)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = default_checkpoint_dir(args.dataset, args.model, args.seed)

    print(args)
    print("JAX devices:", jax.devices())

    train_images, train_labels, train_saliency, test_images, test_labels = load_dataset_with_saliency(args.dataset, args.data_dir)

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    steps_per_epoch = (len(train_images) + args.batch_size - 1) // args.batch_size
    state = create_train_state(
        key=init_key,
        dataset=args.dataset,
        model_name=args.model,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        steps_per_epoch=steps_per_epoch,
        num_classes=num_classes,
    )

    best_test_acc = 0.0
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args.log_dir = os.path.abspath(args.log_dir)
    args.figure_dir = os.path.abspath(args.figure_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    csv_path = os.path.join(args.log_dir, f"{args.dataset}_{args.model}_seed{args.seed}.csv")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch", "dataset", "model",
            "train_loss", "train_acc", "test_loss", "test_acc", "test_error",
            "best_test_acc", "mean_lam", "epoch_time_sec", "total_time_sec",
            "seed", "data_augmentation", "learning_rate", "batch_size", "beta", "salmix_prob",
        ],
    )
    csv_writer.writeheader()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"logging to: {csv_path}")
    print(f"figures will be saved to: {args.figure_dir}")

    train_total = (len(train_images) + args.batch_size - 1) // args.batch_size
    test_total = (len(test_images) + args.batch_size - 1) // args.batch_size

    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_error", "best_test_acc", "mean_lam"]}
    total_start_time = time.time()

    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_losses, train_accs, train_lams = [], [], []
            train_iter = batch_iterator(train_images, train_labels, train_saliency, batch_size=args.batch_size, shuffle=True)
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
                    pad_value=pad_value,
                    num_classes=num_classes,
                )

                loss, acc, lam = float(metrics["loss"]), float(metrics["acc"]), float(metrics["lam"])
                train_losses.append(loss)
                train_accs.append(acc)
                train_lams.append(lam)

                progress.set_postfix(loss=f"{np.mean(train_losses):.4f}", acc=f"{np.mean(train_accs) * 100:.2f}", lam=f"{np.mean(train_lams):.3f}")

            test_losses, test_accs = [], []
            test_iter = batch_iterator(test_images, test_labels, saliency_maps=None, batch_size=args.batch_size, shuffle=False)

            for batch in tqdm(test_iter, total=test_total):
                metrics = eval_step(state=state, batch=batch, num_classes=num_classes)
                test_losses.append(float(metrics["loss"]))
                test_accs.append(float(metrics["acc"]))

            train_loss = float(np.mean(train_losses))
            train_acc = float(np.mean(train_accs) * 100.0)
            mean_lam = float(np.mean(train_lams))
            test_loss = float(np.mean(test_losses))
            test_acc = float(np.mean(test_accs) * 100.0)
            test_error = 100.0 - test_acc

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - total_start_time

            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}, test_error={test_error:.2f}, "
                f"epoch_time≈{epoch_time / 60:.2f} min, total_time≈{total_time / 60:.2f} min"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                checkpointer.save(args.checkpoint_dir, state, force=True)
                print(f"saved best checkpoint to {args.checkpoint_dir}")

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            history["test_error"].append(test_error)
            history["best_test_acc"].append(float(best_test_acc))
            history["mean_lam"].append(mean_lam)

            csv_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "dataset": args.dataset,
                    "model": args.model,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_error": test_error,
                    "best_test_acc": float(best_test_acc),
                    "mean_lam": mean_lam,
                    "epoch_time_sec": float(epoch_time),
                    "total_time_sec": float(total_time),
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

    if len(history["epoch"]) > 0:
        save_training_plots(history, args.figure_dir, args.dataset, args.model, args.seed)

    final_total_time = time.time() - total_start_time
    print(f"Training finished. Approx total time: {final_total_time / 60:.2f} min ({final_total_time:.1f} sec).")


if __name__ == "__main__":
    main()
