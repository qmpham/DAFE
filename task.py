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
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer, var_spec

def update(v,g,lr=1.0):
  if isinstance(g, tf.IndexedSlices):
    return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
  else:
    return v-lr*g

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
    tf.print("source_inputs", source_inputs)
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
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

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
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        print(v.name)
        variables.append(v)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        variables.append(v)
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

def meta_train_v7(config,
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
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
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
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      shared_gradients = []
      adap_gradients = []
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      variables = adap_variables + shared_variables
      adap_variables_name = [v.name for v in adap_variables]
      shared_variables_name = [v.name for v in shared_variables]
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 
      var_spec(variables)
      var_spec(shared_variables)
      var_spec(adap_variables)
      for g,v in zip(gradients, variables):
        if v.name in shared_variables_name:
          shared_gradients.append(g)
        elif v.name in adap_variables_name:
          adap_gradients.append(g)

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(shared_gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(shared_gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      adap_gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      shared_gradients = tape.gradient(meta_training_loss, shared_variables)
      gradients = adap_gradients + shared_gradients
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
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
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
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

def meta_train_v3(config,
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
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
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
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=adap_variables)
      gradients = tape.gradient(training_loss, adap_variables)    
      meta_train_lr = config.get("meta_train_lr", 0.1)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr 
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, adap_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, adap_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables      
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
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
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
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

def meta_train_v5(config,
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
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
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
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
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
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
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

def meta_train_v6(config,
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
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
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
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=shared_variables)
      gradients.extend(tape.gradient(meta_training_loss, shared_variables))
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
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
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
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
    num_examples = tf.reduce_sum(target["length"])
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
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
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
  _number_examples = []
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
      #print("number_examples_in_an_iteration_per_replica: %d"%num_examples)
      _step()
      _loss.append(loss)
      _number_examples.append(num_examples)
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
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
    
def meta_train_v8(config,
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
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
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
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
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
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
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
