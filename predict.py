
import tensorflow as tf
import numpy as np
import re,string,pickle
from tensorflow.keras.layers import TextVectorization
from model import PositionalEmbedding,TransformerDecoder,TransformerEncoder
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


max_decoded_sentence_length = 20
vocab_size = 15000
sequence_length = 20


en = pickle.load(open("eng_vocabulary.pkl", "rb"))
fi = pickle.load(open("fin_vocabulary.pkl", "rb"))
transformer = tf.keras.models.load_model("eng-fin.h5", custom_objects={"PositionalEmbedding": PositionalEmbedding,
                                                                        "TransformerEncoder":TransformerEncoder,
                                                                        "TransformerDecoder":TransformerDecoder})
eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)
fin_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)


eng_vectorization.set_vocabulary(en)
fin_vectorization.set_vocabulary(fi)
fin_vocab = fin_vectorization.get_vocabulary()
fin_index_lookup = dict(zip(range(len(fin_vocab)),fin_vocab))
 
 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
                                    
def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = fin_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = fin_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence



translated = decode_sequence("hi")
print(translated)