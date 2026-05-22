import jax
import jax.numpy as jnp


def saliencymix_batch(
    key,
    images,
    labels,
    saliency_maps,
    beta=1.0,
    salmix_prob=0.5,
):
    """
    images: [B, H, W, C]
    labels: [B]
    saliency_maps: [B, H, W]
    """

    B, H, W, C = images.shape

    key_apply, key_lam, key_perm = jax.random.split(key, 3)

    apply_mix = jax.random.uniform(key_apply) < salmix_prob

    lam = jax.random.beta(key_lam, beta, beta)

    perm = jax.random.permutation(key_perm, B)

    images_perm = images[perm]
    labels_perm = labels[perm]
    saliency_perm = saliency_maps[perm]

    # 对齐原 PyTorch 逻辑：整个 batch 用 shuffled batch 第一张图决定 bbox
    saliency_one = saliency_perm[0]  # [H, W]

    cut_rate = jnp.sqrt(1.0 - lam)

    cut_w = (W * cut_rate).astype(jnp.int32)
    cut_h = (H * cut_rate).astype(jnp.int32)

    flat_idx = jnp.argmax(saliency_one.reshape(-1))

    center_y = flat_idx // W
    center_x = flat_idx % W

    y1 = jnp.clip(center_y - cut_h // 2, 0, H)
    y2 = jnp.clip(center_y + cut_h // 2, 0, H)

    x1 = jnp.clip(center_x - cut_w // 2, 0, W)
    x2 = jnp.clip(center_x + cut_w // 2, 0, W)

    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]

    mask = (
        (yy >= y1)
        & (yy < y2)
        & (xx >= x1)
        & (xx < x2)
    )

    mask = mask[None, :, :, None]  # [1, H, W, 1]

    # 原 PyTorch 是把 shuffled image 的 patch 贴到原 images 上
    mixed = jnp.where(mask, images_perm, images)

    patch_area = (y2 - y1) * (x2 - x1)

    lam_adjusted = 1.0 - patch_area / (H * W)

    mixed_images = jnp.where(apply_mix, mixed, images)

    labels_a = labels
    labels_b = jnp.where(apply_mix, labels_perm, labels)

    final_lam = jnp.where(apply_mix, lam_adjusted, 1.0)

    return mixed_images, labels_a, labels_b, final_lam