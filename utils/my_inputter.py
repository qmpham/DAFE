from opennmt.inputters.text_inputter import WordEmbedder, _get_field, TextInputter
import tensorflow as tf
from opennmt import inputters
from opennmt.models.sequence_to_sequence import _shift_target_sequence
from opennmt.data import text
from opennmt.data import dataset as dataset_util
from opennmt.utils import misc
from utils.utils_ import make_domain_mask
from opennmt.inputters.text_inputter import load_pretrained_embeddings
from opennmt.layers import common

class My_inputter(TextInputter):
    def __init__(self, embedding_size=None, dropout=0.0, **kwargs):        
        super(My_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout

    def forward_fn(self, features, args_dict, training=None):
        embedding = args_dict[self.embedding.name]
        #print("where are we? ________________",embedding)
        outputs = tf.nn.embedding_lookup(embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
        return outputs

    def initialize(self, data_config, asset_prefix=""):
        super(My_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)

    def build(self, input_shape):
        if self.embedding_file:
            pretrained = load_pretrained_embeddings(
                self.embedding_file,
                self.vocabulary_file,
                num_oov_buckets=self.num_oov_buckets,
                with_header=self.embedding_file_with_header,
                case_insensitive_embeddings=self.case_insensitive_embeddings)
            self.embedding_size = pretrained.shape[-1]
            initializer = tf.constant_initializer(value=pretrained.astype(self.dtype))
        else:
            initializer = None
            scope_name = self.name_scope()
            self.embedding = self.add_weight(
                "%s_embedding"%scope_name,
                [self.vocabulary_size, self.embedding_size],
                initializer=initializer,
                trainable=self.trainable)
        super(My_inputter, self).build(input_shape)

    def call(self, features, training=None):
      outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
      outputs = common.dropout(outputs, self.dropout, training=training)
      return outputs

    def make_features(self, element=None, features=None, domain=1, training=None):
        features = super(My_inputter, self).make_features(
            element=element, features=features, training=training)

        if "ids" in features and "domain" in features:
          return features

        features["ids"] = self.tokens_to_ids.lookup(features["tokens"])
        features["domain"] = tf.constant(domain)

        return features
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset
    
    def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            domain=1,
                            batch_type="examples",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            length_bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_training_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=True)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=True)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            length_bucket_width=length_bucket_width,
            single_pass=single_pass,
            process_fn=map_func,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            features_length_fn=self.get_length,
            batch_size_multiple=batch_size_multiple,
            num_shards=num_shards,
            shard_index=shard_index))
        return dataset
    
    def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              domain,
                              num_threads=1,
                              prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_evaluation_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=False)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class LDR_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, num_units=512 , num_domains=6, num_domain_units=8, dropout=0.0, **kwargs):        
        super(LDR_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.fusion_layer = tf.keras.layers.Dense(num_units, use_bias=False)
        self.domain_mask = make_domain_mask(num_domains, num_domains * num_domain_units, num_domain_units=num_domain_units)
        self.num_domain_units = num_domain_units
        self.num_domains = num_domains

    def initialize(self, data_config, asset_prefix=""):
        super(LDR_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    
    def make_features(self, element=None, features=None, domain=1, training=None):
        features = super(LDR_inputter, self).make_features(
            element=element, features=features, training=training)
        if "domain" in features:
            return features
        features["domain"] = tf.constant(domain)

        return features
    
    def build(self, input_shape):
        self.ldr_embed = self.add_weight(
                                "domain_embedding",
                                [self.vocabulary_size, self.num_domain_units * self.num_domains],
                                initializer=None,
                                trainable=True)
        super(LDR_inputter, self).build(input_shape)

    def call(self, features, domain=None, training=None):
        outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
        
        if domain==None:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["ids"])
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,1), (1,tf.shape(outputs)[1],1))
        else:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["ids"])
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,0), (tf.shape(outputs)[0],1))
        
        if domain==None:
            domain_mask = tf.nn.embedding_lookup(self.domain_mask, features["domain"])
            domain_mask = tf.broadcast_to(tf.expand_dims(domain_mask,1),tf.shape(outputs))
        else:
            domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
            domain_mask = tf.broadcast_to(tf.expand_dims(domain_mask,0),tf.shape(outputs))
        print("domain_mask", domain_mask)
        print("ldr_inputs", ldr_inputs)
        ldr_inputs = ldr_inputs * domain_mask
        outputs = tf.concat([outputs, ldr_inputs],-1)
        return outputs
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class DC_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, num_units=512 , num_domains=6, num_domain_units=8, dropout=0.0, **kwargs):        
        super(DC_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.num_domain_units = num_domain_units
        self.num_domains = num_domains

    def initialize(self, data_config, asset_prefix=""):
        super(DC_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    
    def make_features(self, element=None, features=None, domain=1, training=None):
        features = super(DC_inputter, self).make_features(
            element=element, features=features, training=training)
        if "domain" in features:
            return features
        features["domain"] = tf.constant(domain)

        return features
    
    def call(self, features, domain=None, training=None):
        outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
        if domain==None:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["domain"])
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,1), (1,tf.shape(outputs)[1],1))
        else:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, domain)
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,0), (tf.shape(outputs)[0],1))
        outputs = tf.concat([outputs, ldr_inputs],-1)
        return outputs
    
    def build(self, input_shape):
        self.ldr_embed = self.add_weight(
                                "domain_embedding",
                                [self.num_domains, self.num_domain_units],
                                initializer=None,
                                trainable=True)
        super(DC_inputter, self).build(input_shape)
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class Multi_domain_SequenceToSequenceInputter(inputters.ExampleInputter):
    def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
        super(Multi_domain_SequenceToSequenceInputter, self).__init__(
            features_inputter, labels_inputter, share_parameters=share_parameters)
        self.alignment_file = None

    def initialize(self, data_config, asset_prefix=""):
        super(Multi_domain_SequenceToSequenceInputter, self).initialize(data_config, asset_prefix=asset_prefix)
        self.alignment_file = data_config.get("train_alignments")

    def make_dataset(self, data_file, training=None):
        dataset = super(Multi_domain_SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
        if self.alignment_file is None or not training:
            return dataset
        return tf.data.Dataset.zip((dataset, tf.data.TextLineDataset(self.alignment_file)))

    def make_features(self, element=None, features=None, domain=1, training=None):
        if training and self.alignment_file is not None:
            element, alignment = element
        else:
            alignment = None
        features, labels = super(Multi_domain_SequenceToSequenceInputter, self).make_features(
            element=element, features=features, training=training)
        if alignment is not None:
            labels["alignment"] = text.alignment_matrix_from_pharaoh(
                alignment,
                self.features_inputter.get_length(features),
                self.labels_inputter.get_length(labels))
        _shift_target_sequence(labels)
        if "noisy_ids" in labels:
            _shift_target_sequence(labels, prefix="noisy_")
        features["domain"] = tf.constant(domain)
        labels["domain"] = tf.constant(domain)
        return features, labels

    def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             domain,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
        return self.features_inputter.make_inference_dataset(
            features_file,
            batch_size,
            domain=domain,
            length_bucket_width=length_bucket_width,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)

    def make_evaluation_dataset(self,
                                features_file,
                                labels_file,
                                batch_size,
                                domain,
                                num_threads=1,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, domain=domain, training=False)
        dataset = self.make_dataset([features_file, labels_file], training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

    def make_training_dataset(self,
                                features_file,
                                labels_file,
                                batch_size,
                                domain,
                                batch_type="examples",
                                batch_multiplier=1,
                                batch_size_multiple=1,
                                shuffle_buffer_size=None,
                                length_bucket_width=None,
                                maximum_features_length=None,
                                maximum_labels_length=None,
                                single_pass=False,
                                num_shards=1,
                                shard_index=0,
                                num_threads=4,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, domain=domain, training=True)
        dataset = self.make_dataset([features_file, labels_file], training=True)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            features_length_fn=self.features_inputter.get_length,
            labels_length_fn=self.labels_inputter.get_length,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

