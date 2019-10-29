"""This example demonstrates how to train a standard Transformer model using
OpenNMT-tf as a library in about 200 lines of code.
The purpose of this example is to showcase selected OpenNMT-tf APIs that can be
useful in other projects:
* efficient training dataset (with shuffling, bucketing, batching, prefetching, etc.)
* inputter/encoder/decoder API
* dynamic decoding API
Producing a SOTA model is NOT a goal: this usually requires extra steps such as
training a bigger model, using a larger batch size via multi GPU training and/or
gradient accumulation, etc.
"""

import argparse
import logging

import tensorflow as tf
import tensorflow_addons as tfa

import opennmt as onmt
import io
from opennmt import START_OF_SENTENCE_ID
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.misc import print_bytes
from opennmt.data import dataset as dataset_util
from opennmt.optimizers import utils as optimizer_util
tf.get_logger().setLevel(logging.INFO)
from utils.my_inputter import My_inputter
from model import Multi_domain_SequenceToSequence
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder
from decoders.self_attention_decoder import Multi_domain_SelfAttentionDecoder
# Define the model. For the purpose of this example, the model components
# (encoder, decoder, etc.) will be called separately.
devices = tf.config.experimental.list_logical_devices(device_type="GPU")
print(devices)
strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
import numpy as np

def merge_map_fn(*args):
  num = len(args)
  print("args num: ", num)
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
    if tf.rank(src_batches[0][feature])==1:
      print(feature)
      src_batch.get(feature, tf.concat([b[feature] for b in src_batches],0))
    elif tf.rank(src_batches[0][feature])==2:
      print(feature)
      len_max = tf.reduce_max(tf.concat([tf.shape(batch[feature])[1] for batch in src_batches]))
      if src_batches[0][feature].dtype == tf.string:
        src_batch.get(feature, tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], len_max-tf.shape(batch[feature])[1]],"")],0) for batch in src_batches],0))
      else:
        src_batch.get(feature, tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], len_max-tf.shape(batch[feature])[1]],0)],0) for batch in src_batches],0))
    
  for feature in list(tgt_batches[0].keys()):
    if tf.rank(tgt_batches[0][feature])==1:
      print(feature)
      tgt_batch.get(feature, tf.concat([b[feature] for b in tgt_batches],0))
    elif tf.rank(tgt_batches[0][feature])==2:
      print(feature)
      len_max = tf.reduce_max(tf.concat([tf.shape(batch[feature])[1] for batch in tgt_batches]))
      if tgt_batches[0][feature].dtype == tf.string:
        tgt_batch.get(feature, tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], len_max-tf.shape(batch[feature])[1]],"")],0) for batch in tgt_batches],0))
      else:
        tgt_batch.get(feature, tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], len_max-tf.shape(batch[feature])[1]],0)],0) for batch in tgt_batches],0))

  return (src_batch, tgt_batch)

def train(source_file,
          target_file,
          optimizer,
          gradient_accumulator,
          learning_rate,
          model,    
          checkpoint_manager,
          maximum_length=10,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=300,
          save_every=100,
          report_every=100): 
  batch_size = 30
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  print("There are %d in-domain corpora"%len(source_file))
  for src,tgt in zip(source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(
    src,
    tgt,
    batch_size=batch_size,
    batch_type="tokens",
    shuffle_buffer_size=shuffle_buffer_size,
    length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
    maximum_features_length=maximum_length,
    maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(
    src,
    tgt,
    batch_size= batch_size//len(source_file),
    batch_type="tokens",
    shuffle_buffer_size=shuffle_buffer_size,
    length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
    maximum_features_length=maximum_length,
    maximum_labels_length=maximum_length))
  
  meta_train_dataset = tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
    return loss

  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_iteration(next_fn):    
    with strategy.scope():
      #per_replica_source, per_replica_target = next_fn()
      return next_fn()#per_replica_source, per_replica_target

  #meta_train_data_flow = iter(_meta_train_iteration())
  meta_test_data_flow = iter(_meta_test_iteration())
  #print(next(meta_train_data_flow))
  print(next(meta_test_data_flow))
  """
  meta_test_data_flows = [] 
  for meta_test_dataset in meta_test_datasets:
    @dataset_util.function_on_next(meta_test_dataset)
    def _meta_test_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      return loss
    meta_test_data_flows.append(iter(_meta_test_forward()))
  """
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
  
  # Runs the training loop.
  import time
  start = time.time()  
  #meta_train_data_flow = iter(_meta_train_forward())
  #meta_test_data_flow = iter(_meta_test_forward())

  """
  while True:
    #####Training batch
    loss = next(meta_train_data_flow)    
    #print(".....var numb: ", len(model.trainable_variables))
    snapshots = [v.value() for v in model.trainable_variables]
    #print("model: ", model.trainable_variables[3])
    #print("snapshot: ", snapshots[3])    
    _step()
    # print("model: ", model.trainable_variables[3])
    # print("snapshot: ", snapshots[3])
    #####Testing batch
    for meta_test_data_flow in meta_test_data_flows:
      loss = next(meta_test_data_flow)
    weight_reset(snapshots)
    # print("model: ", model.trainable_variables[3])
    # print("snapshot: ", snapshots[3])
    _step()
    ####
    step = optimizer.iterations.numpy()
    if step % report_every == 0:
      elapsed = time.time() - start
      tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
          step, learning_rate(step), loss, elapsed)
      start = time.time()
    if step % save_every == 0:
      tf.get_logger().info("Saving checkpoint for step %d", step)
      checkpoint_manager.save(checkpoint_number=step)
    if step // 2 > train_steps:
      break
  """


def translate(source_file,
              model,
              output_file,
              batch_size=32,
              beam_size=5):
  
  # Create the inference dataset.
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)

    # Run the encoder.
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    decoded = model.decoder.dynamic_decode(
        lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.reshape(tf.shape(ids)[0],[1]), source["domain"][0]), tf.int64)],
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=200)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  while True:
    with io.open(output_file, encoding="utf-8", mode="w") as stream:
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, stream=stream)
          print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "translate"],
                      help="Run type.")
                      
  parser.add_argument("--src", nargs='+', required=True,
                      help="Path to the source file.")
  parser.add_argument("--tgt", nargs='+',
                      help="Path to the target file.")
  parser.add_argument("--output_file", nargs='+',
                      help="Path to the output file.")
  parser.add_argument("--src_vocab", required=True,
                      help="Path to the source vocabulary.")
  parser.add_argument("--tgt_vocab", required=True,
                      help="Path to the target vocabulary.")
  parser.add_argument("--model_dir", default="checkpoint",
                      help="Directory where checkpoint are written.")
  args = parser.parse_args()

  data_config = {
      "source_vocabulary": args.src_vocab,
      "target_vocabulary": args.tgt_vocab
  }

  with strategy.scope():
    
    model = Multi_domain_SequenceToSequence(
      source_inputter=My_inputter(embedding_size=512, domain=1),
      target_inputter=My_inputter(embedding_size=512, domain=1),
      encoder= Multi_domain_SelfAttentionEncoder(
          num_layers=6,
          num_units=512,
          num_heads=8,
          ffn_inner_dim=2048,
          dropout=0.1,
          attention_dropout=0.1,
          ffn_dropout=0.1),
      decoder= Multi_domain_SelfAttentionDecoder(
          num_layers=6,
          num_domains=6,
          num_units=512,
          num_heads=8,
          ffn_inner_dim=2048,
          dropout=0.1,
          attention_dropout=0.1,
          ffn_dropout=0.1))
    """
    model = onmt.models.SequenceToSequence(
    source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
    """
    learning_rate = onmt.schedules.NoamDecay(scale=2.0, model_dim=512, warmup_steps=8000)
    optimizer = tfa.optimizers.LazyAdam(learning_rate)            
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    model.initialize(data_config)
    model.build(None)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.model_dir, max_to_keep=5)
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  
  if args.run == "train":
    train(args.src, args.tgt, optimizer, gradient_accumulator, learning_rate, model, checkpoint_manager)
  elif args.run == "translate":
    translate(args.src, model, args.output_file)
  
if __name__ == "__main__":
  main()
