import jax.numpy as jnp

def integrate(x, y):
    #integrate y(x) over x
    return jnp.trapz(y, x)

def sigmoid(x, c=1.):
    return 1./(1.+jnp.exp(-c*x))

def relu(x, c=1.):
    return jnp.minimum(1., jnp.maximum(0., c*x))


