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

def create_meta_trainining_dataset(strategy, model, domain, source_file, target_file, batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length):
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  for i, src,tgt in zip(domain,source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_meta_train_size,
              batch_type=batch_type,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size= batch_meta_test_size//len(source_file),
              batch_type=batch_type,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  meta_train_dataset = tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)
  def meta_train_fn(input_context):
    #batch_size = input_context.get_per_replica_batch_size(batch_meta_train_size)
    return meta_train_dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id)
  with strategy.scope():
    #meta_train_dataset = strategy.experimental_distribute_dataset(meta_train_dataset)
    meta_test_dataset = strategy.experimental_distribute_datasets_from_function(meta_train_fn)
    meta_test_dataset = strategy.experimental_distribute_dataset(meta_test_dataset)

  return meta_train_dataset, meta_test_dataset