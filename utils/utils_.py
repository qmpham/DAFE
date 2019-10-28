import tensorflow as tf
import numpy as np

def make_domain_mask(num_domains, num_domain_units=8):
  M = np.zeros((num_domains, num_domains * num_domain_units))
  for i in range(num_domains):
    for j in range(i*num_domain_units, (i+1)*num_domain_units):
      M[i,j] = 1
  return tf.constant(M)
