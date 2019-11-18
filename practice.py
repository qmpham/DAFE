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
import sys
import opennmt as onmt
import io
import os
import utils
from opennmt import START_OF_SENTENCE_ID
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.misc import print_bytes
from opennmt.data import dataset as dataset_util
from opennmt.optimizers import utils as optimizer_util
tf.get_logger().setLevel(logging.INFO)
from utils.my_inputter import My_inputter, LDR_inputter
from opennmt.models.sequence_to_sequence import SequenceToSequence
from model import Multi_domain_SequenceToSequence, LDR_SequenceToSequence
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder
from decoders.self_attention_decoder import Multi_domain_SelfAttentionDecoder
import numpy as np
from utils.dataprocess import merge_map_fn, create_meta_trainining_dataset, create_trainining_dataset, create_multi_domain_meta_trainining_dataset
from opennmt.utils import BLEUScorer
from opennmt.inputters.text_inputter import WordEmbedder
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer

def debug(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables   
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = tf.gradients(training_loss, variables)
    ##### Inner adaptation
    args_dict = dict()
    def update(v,g,lr=0.01):
      if "embedding" in v.name:
        return tf.tensor_scatter_nd_sub(v/lr,g.indices,g)*lr
      else:
        return v - lr* g
    for g, v in zip(gradients, variables):
      args_dict.update({v.name:update(v,g)})
    #### Meta_loss:
    outputs, _ = model.forward_fn(meta_test_source,
        args_dict,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(meta_test_target["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  
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
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  #meta_test_data_flow = iter(_meta_test_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      _loss.append(loss)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      #print("number_examples_per_replica: ", num_examples)

def meta_train_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    meta_train_gradient_accumulator = optimizer_util.GradientAccumulator()  
    meta_test_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_meta_train_gradients(source, target):
    print("source: ", source)
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
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_train_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_meta_test_gradients(source, target):
    print("source: ", source)
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
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_test_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_meta_train_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_train_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_train_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_train_gradient_accumulator.reset()

  def _apply_meta_test_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_test_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_test_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_test_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_meta_train_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, _ = strategy.experimental_run_v2(
          _accumulate_meta_test_gradients, args=(per_replica_source, per_replica_target))
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
  def _meta_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_train_gradients)

  @tf.function
  def _meta_test_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_test_gradients)

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
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  meta_test_data_flow = iter(_meta_test_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      #print("number_examples_per_replica: ", num_examples)
      _loss.append(loss)  
      #snapshots = [v.value() for v in model.trainable_variables]
      _meta_train_step()
      #####Testing batch
      loss = next(meta_test_data_flow)
      #weight_reset(snapshots)
      _meta_test_step()
      ####      
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
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    #model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    reported_loss = loss[0] / loss[2]
    variables = model.trainable_variables       
    args_dict = dict()
    for v in variables:
      args_dict.update({v.name:v})
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = tf.gradients(training_loss, variables)
    ##### Inner adaptation
    def update(v,g,lr=1.0):
      if "embedding" in v.name:
        # print("embedding gradient's values: __________", g.values)
        # print("embedding gradient's indices: _________", g.indices)
        return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
      else:
        return v - lr*g
    for g, v in zip(gradients, variables):
      if config.get("stopping_gradient",True):
        args_dict.update({v.name: update(v,tf.stop_gradient(g))})
      else:
        args_dict.update({v.name: update(v,g)})
    
    #### Meta_loss:
    print("number variables: ", len(list(args_dict.keys())))  
    outputs, _ = model.forward_fn(meta_test_source,
        args_dict,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    meta_training_loss = loss[0] / loss[1]
    meta_reported_loss = loss[0] / loss[2]
    meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
    gradients = optimizer.get_gradients(meta_training_loss, variables)
    #for g in gradients:
    #  print(g)
    gradient_accumulator(gradients)
    num_examples = tf.shape(meta_test_target["length"])[0]
    return meta_reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      _loss.append(loss)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def finetuning(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length)
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
    variables = []
    for var in model.trainable_variables:
      if "ldr_embedding" in var.name or "ldr_inputter" in var.name:
        print(var.name)
        variables.append(var)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = []
    for var in model.trainable_variables:
      if "ldr_embedding" in var.name or "ldr_inputter" in var.name:
        variables.append(var)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _finetuning_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  finetuning_data_flow = iter(_finetuning_forward())
  
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(finetuning_data_flow)
      _step()     
      _loss.append(loss)
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

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
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length)
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
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
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
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

        
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, num_examples = next(train_data_flow)    
      print("number_examples_in_an_iteration_per_replica: %d"%num_examples)
      _step()
      _loss.append(loss)
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break
    
def translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              length_penalty,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=32,
              beam_size=5):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", checkpoint_manager.latest_checkpoint)
  print("In domain %d"%domain)
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
    if experiment=="residual":
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment=="residual":
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment=="ldr":
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain)
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
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
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    return scorer(reference, output_file)
  
def main():
  devices = tf.config.experimental.list_logical_devices(device_type="GPU")
  print(devices)
  strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train", "translate", "debug","metatrainv1", "metatrainv2", "finetune"], help="Run type.")
  parser.add_argument("--config", required=True , help="configuration file")
  parser.add_argument("--src")
  parser.add_argument("--output")
  parser.add_argument("--domain")
  parser.add_argument("--ref", default=None)
  args = parser.parse_args()
  print("Running mode: ", args.run)
  config_file = args.config
  with open(config_file, "r") as stream:
      config = yaml.load(stream)
  if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))
  data_config = {
      "source_vocabulary": config["src_vocab"],
      "target_vocabulary": config["tgt_vocab"]
  }
  experiment = config.get("experiment","residual")
  print("running experiment: ", experiment)
  if experiment=="residual":
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
  elif experiment=="ldr":
    model = LDR_SequenceToSequence(
    source_inputter=LDR_inputter(embedding_size=464),
    target_inputter=LDR_inputter(embedding_size=464),
    encoder=onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="baseline":
    model = SequenceToSequence(
    source_inputter=WordEmbedder(embedding_size=512),
    target_inputter=WordEmbedder(embedding_size=512),
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
  elif experiment=="pretrain":
    return
    
  learning_rate = onmt.schedules.ScheduleWrapper(schedule=onmt.schedules.NoamDecay(scale=1.0, model_dim=512, warmup_steps=4000), step_duration= config.get("step_duration",16))
  meta_train_optimizer = tfa.optimizers.LazyAdam(1.0)
  meta_test_optimizer = tfa.optimizers.LazyAdam(learning_rate)
  checkpoint = tf.train.Checkpoint(model=model, optimizer=meta_test_optimizer)   
  model.initialize(data_config)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, config["model_dir"], max_to_keep=5)
  ######
  model.params.update({"label_smoothing": 0.1})
  model.params.update({"average_loss_in_time": True})
  model.params.update({"beam_width": 5})
  ######
  if args.run == "metatrainv2":
    meta_train_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv1":
    meta_train_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run =="train":
    train(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "translate":
    model.build(None)
    print("translate in domain %d"%(int(args.domain)))
    translate(args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.domain), args.output, length_penalty=0.6, experiment=experiment)
  elif args.run == "finetune":
    finetuning(config, meta_train_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "debug":
    debug(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
if __name__ == "__main__":
  main()
