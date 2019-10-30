"""Dataset creation and transformations."""

import numpy as np
import tensorflow as tf

def merge_map_fn(*args):
  
  src_batches = []
  tgt_batches = []
  for (src,tgt) in args:
    src_batches.append(src)
    tgt_batches.append(tgt)
  print("element numb: ",len(src_batches))
  src_batch = {}
  tgt_batch = {}
  print(src_batches[0].keys())
  for feature in list(src_batches[0].keys()):
    if feature!="ids" and feature!="tokens":
      print(feature, src_batches[0][feature])
      src_batch.update({feature: tf.concat([b[feature] for b in src_batches],0)})
    else:
      print(feature, src_batches[0][feature])
      len_max = tf.reduce_max([tf.shape(batch[feature])[1] for batch in src_batches])
      if src_batches[0][feature].dtype == tf.string:
        src_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],"")],1) for batch in src_batches],0)})
      else:
        src_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.cast(tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],0),tf.int64)],1) for batch in src_batches],0)})
    
  for feature in list(tgt_batches[0].keys()):
    if feature!="ids" and feature!="tokens" and feature!="ids_out":
      print(feature, tgt_batches[0][feature])
      tgt_batch.update({feature: tf.concat([b[feature] for b in tgt_batches],0)})    
    else:
      print(feature, tgt_batches[0][feature])
      len_max = tf.reduce_max([tf.shape(batch[feature])[1] for batch in tgt_batches])
      if tgt_batches[0][feature].dtype == tf.string:
        tgt_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],"")],1) for batch in tgt_batches],0)})
      else:
        tgt_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.cast(tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],0),tf.int64)],1) for batch in tgt_batches],0)})
  print(src_batch,tgt_batch)
  return src_batch, tgt_batch