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

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_trainining_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    ##### squeeze first dimension of per replica batch
    """
    for k in list(meta_train_source.keys()):
      batch = meta_train_source[k]
      meta_train_source.update({k:tf.squeeze(batch,0)})
    for k in list(meta_train_target.keys()):
      batch = meta_train_target[k]
      meta_train_target.update({k:tf.squeeze(batch,0)})
    for k in list(meta_test_source.keys()):
      batch = meta_test_source[k]
      meta_test_source.update({k:tf.squeeze(batch,0)})
    for k in list(meta_test_target.keys()):
      batch = meta_test_target[k]
      meta_test_target.update({k:tf.squeeze(batch,0)})
    """
    #######
    """
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    reported_loss = loss[0] / loss[2]    
    """
    reported_loss = 0
    num_examples = tf.shape(meta_test_target["ids"])[0]
    tf.print("ids shape: ", tf.shape(meta_test_target["ids"]))
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
  #_loss = []  
  #meta_train_flow = iter(meta_train_dataset)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, num_examples = next(meta_train_data_flow)  
      #_loss.append(loss)
      print("number examples per replica: ", num_examples)
      #print(next(meta_train_flow))

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
  with strategy.scope():
    model.create_variables()
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
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
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    tf.print("meta_train_source: ",meta_train_source, output_stream=sys.stderr)
    tf.print("meta_train_target: ",meta_train_target, output_stream=sys.stderr)
    tf.print("meta_test_source: ",meta_test_source, output_stream=sys.stderr)
    tf.print("meta_test_target: ",meta_test_target, output_stream=sys.stderr)
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
        print(v)
        print(g)
        return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
      else:
        return v-lr*g
    if config.get("stopping_gradient",True):
      print("apply stopping_gradient")
      for g, v in zip(gradients, variables):      
        args_dict.update({v.name: v-g})
    else:
      print("passing gradient")
      for g, v in zip(gradients, variables):
        args_dict.update({v.name: update(v,g)})
    #### Meta_loss:
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
    gradient_accumulator(gradients)
    num_examples = tf.shape(meta_test_target["length"])[0]
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

