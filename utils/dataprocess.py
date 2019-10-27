"""Dataset creation and transformations."""

import numpy as np
import tensorflow as tf


def _get_output_shapes(dataset):

  return tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)

def get_dataset_size(dataset, batch_size=5000):

  if batch_size is not None:
    dataset = dataset.batch(batch_size)

  def _reduce_func(count, element):
    element = tf.nest.flatten(element)[0]
    batch_size = tf.shape(element)[0]
    return count + tf.cast(batch_size, count.dtype)

  return dataset.reduce(tf.constant(0, dtype=tf.int64), _reduce_func)

def filter_irregular_batches(multiple):

  if multiple == 1:
    return lambda dataset: dataset

  def _predicate(*x):
    flat = tf.nest.flatten(x)
    batch_size = tf.shape(flat[0])[0]
    return tf.equal(batch_size % multiple, 0)

  return lambda dataset: dataset.filter(_predicate)

def filter_examples_by_length(maximum_features_length=None,
                              maximum_labels_length=None,
                              features_length_fn=None,
                              labels_length_fn=None):

  if features_length_fn is None and labels_length_fn is None:
    return lambda dataset: dataset

  def _length_constraints(length, maximum_length):
    # Work with lists of lengths which correspond to the general multi source case.
    if not isinstance(length, list):
      length = [length]
    if not isinstance(maximum_length, list):
      maximum_length = [maximum_length]
    # Unset maximum lengths are set to None (i.e. no constraint).
    maximum_length += [None] * (len(length) - len(maximum_length))
    constraints = []
    for l, maxlen in zip(length, maximum_length):
      constraints.append(tf.greater(l, 0))
      if maxlen is not None:
        constraints.append(tf.less_equal(l, maxlen))
    return constraints

  def _predicate(features, labels):
    cond = []
    features_length = features_length_fn(features) if features_length_fn is not None else None
    labels_length = labels_length_fn(labels) if labels_length_fn is not None else None
    if features_length is not None:
      cond.extend(_length_constraints(features_length, maximum_features_length))
    if labels_length is not None:
      cond.extend(_length_constraints(labels_length, maximum_labels_length))
    return tf.reduce_all(cond)

  return lambda dataset: dataset.filter(_predicate)

def random_shard(shard_size, dataset_size):
  
  num_shards = -(-dataset_size // shard_size)  # Ceil division.
  offsets = np.linspace(0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64)

  def _random_shard(dataset):
    sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
    sharded_dataset = sharded_dataset.shuffle(num_shards)
    sharded_dataset = sharded_dataset.flat_map(
        lambda offset: dataset.skip(offset).take(shard_size))
    return sharded_dataset

  return _random_shard

def shuffle_dataset(buffer_size, shuffle_shards=True):
  def _shuffle(dataset):
    sample_size = buffer_size
    if sample_size < 0 or shuffle_shards:
      dataset_size = get_dataset_size(dataset)
      tf.get_logger().info("Training on %d examples", dataset_size)
      if sample_size < 0:
        sample_size = dataset_size
      elif sample_size < dataset_size:
        dataset = dataset.apply(random_shard(sample_size, dataset_size))
    dataset = dataset.shuffle(sample_size)
    return dataset

  return _shuffle

def batch_dataset(batch_size, padded_shapes=None):
  
  return lambda dataset: dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes or _get_output_shapes(dataset))

def batch_sequence_dataset(batch_size,
                           batch_type="examples",
                           batch_multiplier=1,
                           batch_size_multiple=1,
                           length_bucket_width=None,
                           length_fn=None,
                           padded_shapes=None):
  
  batch_size = batch_size * batch_multiplier

  def _get_bucket_id(features, length_fn):
    default_id = tf.constant(0, dtype=tf.int32)
    if length_fn is None:
      return default_id
    lengths = length_fn(features)
    if lengths is None:
      return default_id
    if not isinstance(lengths, list):
      lengths = [lengths]  # Fallback to the general case of parallel inputs.
    lengths = [length // length_bucket_width for length in lengths]
    return tf.reduce_max(lengths)

  def _key_func(*args):
    length_fns = length_fn
    if length_fns is None:
      length_fns = [None for _ in args]
    elif not isinstance(length_fns, (list, tuple)):
      length_fns = [length_fns]
    if len(length_fns) != len(args):
      raise ValueError("%d length functions were passed but this dataset contains "
                       "%d parallel elements" % (len(length_fns), len(args)))
    # Take the highest bucket id.
    bucket_id = tf.reduce_max([
        _get_bucket_id(features, length_fn)
        for features, length_fn in zip(args, length_fns)])
    return tf.cast(bucket_id, tf.int64)

  def _reduce_func(unused_key, dataset):
    return dataset.apply(batch_dataset(batch_size, padded_shapes=padded_shapes))

  def _window_size_func(key):
    if length_bucket_width > 1:
      key += 1  # For length_bucket_width == 1, key 0 is unassigned.
    size = batch_size // (key * length_bucket_width)
    required_multiple = batch_multiplier * batch_size_multiple
    if required_multiple > 1:
      size = size + required_multiple - size % required_multiple
    return tf.cast(tf.maximum(size, required_multiple), tf.int64)

  if length_bucket_width is None:
    return batch_dataset(batch_size, padded_shapes=padded_shapes)

  if batch_type == "examples":
    return tf.data.experimental.group_by_window(
        _key_func, _reduce_func, window_size=batch_size)
  elif batch_type == "tokens":
    return tf.data.experimental.group_by_window(
        _key_func, _reduce_func, window_size_func=_window_size_func)
  else:
    raise ValueError(
        "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))


def training_pipeline(batch_size,
                      batch_type="examples",
                      batch_multiplier=1,
                      batch_size_multiple=1,
                      process_fn=None,
                      length_bucket_width=None,
                      features_length_fn=None,
                      labels_length_fn=None,
                      maximum_features_length=None,
                      maximum_labels_length=None,
                      single_pass=False,
                      num_shards=1,
                      shard_index=0,
                      num_threads=None,
                      shuffle_buffer_size=None,
                      prefetch_buffer_size=None):

  def _pipeline(dataset):
    if num_shards > 1:
      dataset = dataset.shard(num_shards, shard_index)
    if shuffle_buffer_size is not None and shuffle_buffer_size != 0:
      dataset = dataset.apply(shuffle_dataset(shuffle_buffer_size))
    if process_fn is not None:
      dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 4)
    dataset = dataset.apply(filter_examples_by_length(
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        features_length_fn=features_length_fn,
        labels_length_fn=labels_length_fn))
    dataset = dataset.apply(batch_sequence_dataset(
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        batch_size_multiple=batch_size_multiple,
        length_bucket_width=length_bucket_width,
        length_fn=[features_length_fn, labels_length_fn]))
    dataset = dataset.apply(filter_irregular_batches(batch_multiplier))
    if not single_pass:
      dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _pipeline

def inference_pipeline(batch_size,
                       process_fn=None,
                       length_bucket_width=None,
                       length_fn=None,
                       num_threads=None,
                       prefetch_buffer_size=None):

  def _inject_index(index, x):
    x["index"] = index
    return x

  def _pipeline(dataset):
    if process_fn is not None:
      dataset = dataset.map(process_fn, num_parallel_calls=num_threads)
    if length_bucket_width is not None and length_bucket_width > 0:
      if length_fn is None:
        raise ValueError("length_fn is required when reordering by length")
      if not isinstance(_get_output_shapes(dataset), dict):
        raise ValueError("Reordering by length expects dataset elements to be Python dicts")
      dataset = dataset.enumerate()
      dataset = dataset.map(_inject_index)
      dataset = dataset.apply(batch_sequence_dataset(
          batch_size,
          length_bucket_width=length_bucket_width,
          length_fn=length_fn))
    else:
      dataset = dataset.apply(batch_dataset(batch_size))
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _pipeline

def function_on_next(dataset, as_numpy=False):

  def decorator(func):
    def _fun():
      iterator = iter(dataset)

      @tf.function
      def _tf_fun():
        return func(lambda: next(iterator))

      while True:
        try:
          outputs = _tf_fun()
          if as_numpy:
            outputs = tf.nest.map_structure(lambda x: x.numpy(), outputs)
          yield outputs
        except tf.errors.OutOfRangeError:
          break

    return _fun

  return decorator
