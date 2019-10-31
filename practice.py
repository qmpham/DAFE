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
import yaml
import tensorflow as tf
import tensorflow_addons as tfa

import opennmt as onmt
import io
import os
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
import numpy as np
from utils.dataprocess import merge_map_fn, create_meta_trainining_dataset
from opennmt.utils import BLEUScorer

def debug(source_file,
          target_file,
          domain,
          optimizer,
          strategy,          
          learning_rate,
          model,    
          checkpoint_manager,
          maximum_length=80,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          report_every=100):
  batch_size = 100
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  print("There are %d in-domain corpora"%len(source_file))
  for i, src,tgt in zip(domain,source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_size,
              batch_type="tokens",
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size= batch_size//len(source_file),
              batch_type="tokens",
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  meta_train_dataset = tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)

  with strategy.scope():
    meta_train_dataset = strategy.experimental_distribute_dataset(meta_train_dataset)
    meta_test_dataset = strategy.experimental_distribute_dataset(meta_test_dataset)
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

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

  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_forward(next_fn):    
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
      return next_fn()
 
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

  meta_train_data_flow = iter(_meta_train_iteration())
  meta_test_data_flow = iter(_meta_test_iteration())
  print("meta train: ", next(meta_train_data_flow))
  print("meta test: ", next(meta_test_data_flow))


def train(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=100,
          eval_every=100,
          report_every=100): 
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  batch_size = batch_size
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

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

  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_forward(next_fn):    
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
      return next_fn()
 
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
  meta_train_data_flow = iter(_meta_train_forward())
  meta_test_data_flow = iter(_meta_test_forward())
  _loss = []  

  while True:
    #####Training batch
    loss = next(meta_train_data_flow)    
    snapshots = [v.value() for v in model.trainable_variables]
    _step()
    #####Testing batch
    loss = next(meta_test_data_flow)
    weight_reset(snapshots)
    _step()
    ####
    _loss.append(loss)
    step = optimizer.iterations.numpy()//2
    if step % report_every == 0:
      elapsed = time.time() - start
      tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
          step, learning_rate(step), np.mean(_loss), elapsed)
      _loss = []
      start = time.time()
    if step % save_every == 0 and optimizer.iterations.numpy()%2==0:
      tf.get_logger().info("Saving checkpoint for step %d", step)
      checkpoint_manager.save(checkpoint_number=step)
    if step % eval_every == 0 and optimizer.iterations.numpy()%2==0:
      checkpoint_path = checkpoint_manager.latest_checkpoint
      tf.summary.experimental.set_step(step)
      for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
        score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file)
        tf.summary.scalar("BLEU_%d"%i, score, description="BLEU on test set %s"%src)
    if step > train_steps:
      break
  
def translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              batch_size=32,
              beam_size=5):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", checkpoint_manager.latest_checkpoint)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
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
        lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)],
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=200)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  scorer = BLEUScorer()
  print(output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
  return scorer(reference, output_file)

def main():
  devices = tf.config.experimental.list_logical_devices(device_type="GPU")
  print(devices)
  strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "translate", "debug"], help="Run type.")
  parser.add_argument("--config", required=True , help="configuration file")
  args = parser.parse_args()

  config_file = args.config
  with open(config_file, "r") as stream:
      config = yaml.load(stream)
  if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))
  data_config = {
      "source_vocabulary": config["src_vocab"],
      "target_vocabulary": config["tgt_vocab"]
  }

  model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder(
        num_layers=6,
        num_domains=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))

  learning_rate = onmt.schedules.ScheduleWrapper(schedule=onmt.schedules.NoamDecay(scale=1.0, model_dim=512, warmup_steps=4000), step_duration= 16)
  optimizer = tfa.optimizers.LazyAdam(learning_rate)
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)   
  model.initialize(data_config)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, config["model_dir"], max_to_keep=5)
 
  if args.run == "train":
    train(config, optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint)
  elif args.run == "translate":
    model.build(None)
    print(int(config["src_domain"]))
    translate(config["src_test"], config["reference"], model, checkpoint_manager,
              checkpoint, int(config["src_domain"]), config["output_file"])
  elif args.run == "debug":
    debug(config["src"], config["tgt"], config["domain"], optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint)  
  
if __name__ == "__main__":
  main()
