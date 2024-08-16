import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Add

def positional_encoding( max_len, d_model):
    d_model /= 2
    pos = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]/d_model
    frequency = pos * ( 1 / ( 10000 ** dims) )


    pos_encoding = np.concatenate( [np.sin(frequency), np.cos(frequency)], axis=-1)
    return tf.cast( pos_encoding, dtype = tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,vocab_size, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = positional_encoding(max_len, d_model)
        self.embedding = tf.keras.layers.Embedding( vocab_size, d_model, mask_zero = False )

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        max_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt( tf.cast(self.d_model, tf.float32) )
        x = x + self.positional_encoding[ tf.newaxis, :max_length, :]
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dense_hidden_num_units , d_model):
        super().__init__()
        self.ffn = tf.keras.Sequential(
            [
                Dense(dense_hidden_num_units, activation="relu"),
                Dense(d_model)
            ]
        )
        self.add = Add()
        self.layernorm = LayerNormalization()
    
    def call( self, x):
        x = self.add([x, self.ffn(x)])
        x = self.layernorm(x)
        return x

class BaseAttention( tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()
    
class EncoderSelfAttention(BaseAttention):
    def call(self, x):
        mha_out = self.mha( query = x, key = x, value = x)
        x = self.add([x, mha_out])
        x = self.layernorm(x)
        return x

class CrossAttention(BaseAttention):
    def call(self, x, context):
        mha_out = self.mha( query = x, key= context, value = context)
        x = self.add([x, mha_out])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        mha_out = self.mha( query = x, key = x, value = x, use_causal_mask = True)
        x = self.add([x, mha_out])
        x = self.layernorm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dense_hidden_num_units):
        super().__init__()
        self.self_attention = EncoderSelfAttention( num_heads = num_heads, key_dim = d_model)
        self.ffn = FeedForward(dense_hidden_num_units, d_model)
    
    def call(self,x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder( tf.keras.layers.Layer ):
    def __init__(self, num_heads, d_model, num_layers, vocab_size, dense_hidden_num_units, max_len ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model, max_len)
        
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append( EncoderLayer( num_heads, d_model, dense_hidden_num_units ) )
    
    def call(self,x):
        x = self.positional_embedding(x)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dense_hidden_num_units):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention( num_heads = num_heads, key_dim = d_model)
        self.cross_attention = CrossAttention( num_heads = num_heads, key_dim = d_model )
        self.ffn = FeedForward(dense_hidden_num_units, d_model)
    
    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dense_hidden_num_units, vocab_size, max_len):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_embedding = PositionalEmbedding( vocab_size, d_model, max_len )
        self.decoder_layers = []
        for i in range(num_layers):
            self.decoder_layers.append(
                DecoderLayer(d_model, num_heads, dense_hidden_num_units)
            )
    
    def call(self, x, context):
        x = self.positional_embedding(x)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, context)
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dense_hidden_num_units, input_vocab_size, target_vocab_size, inp_max_len, out_max_len):
        super().__init__()
        self.encoder = Encoder( num_heads, d_model, num_layers, input_vocab_size, dense_hidden_num_units, inp_max_len)
        self.decoder = Decoder( num_layers, d_model, num_heads, dense_hidden_num_units, target_vocab_size, out_max_len)

        self.final_layer = Dense(target_vocab_size, activation = "linear")

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        output = self.final_layer(x)
        return output