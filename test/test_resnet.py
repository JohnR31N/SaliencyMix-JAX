import jax
import jax.numpy as jnp

from model.resnet import ResNet18


key = jax.random.PRNGKey(0)

model = ResNet18(num_classes=10)

x = jnp.ones((8, 32, 32, 3))

variables = model.init(key, x, train=True)

logits = model.apply(
    variables,
    x,
    train=True,
    mutable=["batch_stats"],
)

print(type(logits))
print(logits)