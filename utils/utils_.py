import tensorflow as tf
import numpy as np
import subprocess
import io
import re
import os
def make_domain_mask(num_domains, num_domain_units=8, dtype=tf.float32):
  M = np.zeros((num_domains, num_domains * num_domain_units))
  for i in range(num_domains):
    for j in range(i*num_domain_units, (i+1)*num_domain_units):
      M[i,j] = 1
  return tf.constant(M, dtype=dtype)

def masking(ids, mask_id, noise_percentage=0.15):
  return 

def _compute_fans(shape):
  
  if len(shape) < 1:  
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


def variance_scaling_initialier(shape, scale=1.0, mode="fan_in", distribution="uniform"):
  assert mode in ["fan_in","fan_out","fan_avg"]
  assert distribution in ["uniform","truncated_normal","untruncated_normal"]

  fan_in, fan_out = _compute_fans(shape)
  if mode == "fan_in":
    n = fan_in
  elif mode == "fan_out":
    n = fan_out
  else:
    n = (fan_in + fan_out)/2
  
  if distribution == "uniform":
    limit = np.sqrt(3 * scale / n)
    return np.random.uniform(-limit, limit, shape)
  elif distribution == "truncated_normal":
    stddev = np.sqrt(scale / n) 
    return tf.random.truncated_normal(shape, mean=0.0, stddev=stddev)
  else:
    stddev = np.sqrt(scale / n)
    return tf.random.normal(shape, mean=0.0, stddev=stddev)

def var_spec(var):
  print("var inspector:_____")
  print(var)
  if isinstance(var,list):
    print("contains:  %d elements"%len(var))

class MultiBLEUScorer(object):

  def __init__(self, bleu_script="multi-bleu.perl"):
    
    self._bleu_script = bleu_script

  def __call__(self, labels_file, predictions_path):
    utils_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(utils_dir)
    script_path = os.path.join(project_dir, "scripts")
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [os.path.join(script_path, self._bleu_script), labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError:      
      return None



  