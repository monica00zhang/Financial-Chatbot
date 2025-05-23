import tensorflow as tf
from model import load_chat_model
from utils import load_tokenizer, preprocess_sentence
import pandas as pd

MAX_LENGTH = 193

tokenizer_path = "model/test_tokenizer.pkl"
tokenizer = load_tokenizer(tokenizer_path)
print("Tokenizer loaded!")

model_path = "model/test_financial_chatbot_model"
model = load_chat_model(model_path)


def evaluate( sentence):
    sentence = preprocess_sentence(sentence)

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def get_bot_response(user_input):
    prediction = evaluate(user_input)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return predicted_sentence

