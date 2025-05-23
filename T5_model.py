import math
import os

import pandas as pd
import numpy as np
from time import time
import tensorflow as tf
import tensorflow_datasets as tfds


tf.keras.utils.set_random_seed(1234)
print(f"Tensorflow version {tf.__version__}")


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        assert d_model % num_heads == 0
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def build(self, input_shape):
        """
        This method is called when the layer is created and inputs are known.
        Here you can initialize any parameters or state.
        Since we don't need to add additional state, it's left empty.
        """
        pass

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
            }
        )
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(
            lambda inputs: tf.reshape(
                inputs, shape=(batch_size, -1, self.num_heads, self.depth)
            )
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        )(scaled_attention)

        # concatenation of heads
        concat_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.reshape(
                scaled_attention, (batch_size, -1, self.d_model)
            )
        )(scaled_attention)

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # 预计算位置编码矩阵
        position = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]  # (max_seq_len, 1)
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) *
            (-math.log(10000.0) / d_model
        ))  # (d_model//2,)

        # 分别计算sin和cos
        sin_vals = tf.sin(position * div_term)  # (max_seq_len, d_model//2)
        cos_vals = tf.cos(position * div_term)  # (max_seq_len, d_model//2)

        # 交替组合sin和cos
        pe = tf.stack([sin_vals, cos_vals], axis=-1)  # (max_seq_len, d_model//2, 2)
        pe = tf.reshape(pe, [max_seq_len, d_model])  # (max_seq_len, d_model)

        self.pos_encoding = pe[tf.newaxis, :, :]  # (1, max_seq_len, d_model)

    def call(self, inputs):
        # 确保输入是密集张量
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)

        seq_len = tf.shape(inputs)[1]
        output = inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model
        })
        return config



def load_chat_model(filename):
    model = tf.keras.models.load_model(
        filename,
        custom_objects={
            "create_padding_mask": create_padding_mask,
            "create_look_ahead_mask":create_look_ahead_mask,
            "PositionalEncoding": PositionalEncoding,
            "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
        },
        compile=False,
    )
    return model



