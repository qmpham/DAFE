from opennmt.inputters.text_inputter import WordEmbedder, _get_field
import tensorflow as tf

class My_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, dropout=0.0, domain=0, **kwargs):        
        super(My_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.domain = int(domain)

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
    def make_features(self, element=None, features=None, training=None):
        features = super(My_inputter, self).make_features(
            element=element, features=features, training=training)
        if "domain" in features:
            return features
        features["domain"] = tf.constant(self.domain)

        return features
