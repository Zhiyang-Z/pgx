import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial

@partial(jax.pmap, axis_name="i") # axis_name for using collective operations.
def train_step(train_state: train_state.TrainState, batch):
    def loss_fn(params):
        (logits, v), new_model_state = train_state.apply_fn({'params': params, **train_state.model_state},
                                                        batch.obs,
                                                        mutable=list(train_state.model_state.keys()),
                                                        is_training=True)
        policy_loss = optax.softmax_cross_entropy(logits, batch.policy_tgt).mean()
        value_loss = (optax.l2_loss(v, batch.value_tgt) * batch.mask).mean()
        loss = policy_loss + value_loss
        return loss, (policy_loss, value_loss, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, new_model_state)), grads = grad_fn(train_state.params)
    grads = jax.lax.pmean(grads, axis_name="i")
    train_state = train_state.apply_gradients(grads=grads, model_state=new_model_state)
    
    train_info = {
        'loss': loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
    }
    return train_state, train_info


def forward(train_state: train_state.TrainState, x):
    logits, v = train_state.apply_fn({'params': train_state.params, **train_state.model_state},
                                         x,
                                         mutable=False,
                                         is_training=False)
    return logits, v