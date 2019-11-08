import tensorflow as tf
import numpy as np

def make_domain_mask(num_domains, num_domain_units=8, dtype=tf.float32):
  M = np.zeros((num_domains, num_domains * num_domain_units))
  for i in range(num_domains):
    for j in range(i*num_domain_units, (i+1)*num_domain_units):
      M[i,j] = 1
  return tf.constant(M, dtype=dtype)

def masking(ids, mask_id, noise_percentage=0.15):
  return 

def variance_scaling_initialier(shape, scale=1.0, mode="fan_in", distribution="uniform"):
  assert mode in ["fan_in","fan_out","fan_avg"]
  assert distribution in ["uniform","truncated_normal","untruncated_normal"]
  if mode == "fan_in":
    n = shape[0]
  elif mode == "fan_out":
    n = shape[-1]
  else:
    n = np.mean(shape)
  
  if distribution == "uniform":
    limit = np.sqrt(3 * scale / n)
    return np.random.uniform(-limit, limit, shape)
  elif distribution == "truncated_normal":
    stddev = np.sqrt(scale / n) 
    return tf.random.truncated_normal(shape, mean=0.0, stddev=stddev)
  else:
    stddev = np.sqrt(scale / n)
    return tf.random.normal(shape, mean=0.0, stddev=stddev)

@tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(w)

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))
  
  


  