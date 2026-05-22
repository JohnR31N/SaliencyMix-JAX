import jax
import jax.numpy as jnp

from augmentations.saliencymix import saliencymix_batch


key = jax.random.PRNGKey(0)

images = jax.random.normal(key, (8, 32, 32, 3))
labels = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])

# 先用 fake saliency map 测试
saliency_maps = jnp.mean(jnp.abs(images), axis=-1)

mixed_images, labels_a, labels_b, lam = saliencymix_batch(
    key,
    images,
    labels,
    saliency_maps,
    beta=1.0,
    salmix_prob=1.0,
)

print("mixed_images:", mixed_images.shape)
print("labels_a:", labels_a)
print("labels_b:", labels_b)
print("lam:", lam)