import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

class Dataset():
    def __init__(self,inp_dataset_path, out_dataset_path, inp_max_length, out_max_length , batch_size, buffer_size):
        self.df = tf.data.TextLineDataset(inp_dataset_path)
        self.df = self.df.map( self.extract_info )
        self.df = self.df.batch(batch_size)

        self.out_df = tf.data.TextLineDataset(out_dataset_path)
        self.out_df = self.out_df.batch(batch_size)

        self.input_vectorizer = TextVectorization( output_sequence_length=inp_max_length, standardize="strip_punctuation", output_mode="int")
        self.input_vectorizer.adapt( self.df )
        self.output_vectorizer = TextVectorization( output_sequence_length=out_max_length , standardize=None, output_mode="int")
        self.output_vectorizer.adapt( self.out_df )

        self.inp_max_length = inp_max_length
        self.out_max_length = out_max_length
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        # self.count = 0

    # def get_unique_tags(self, df):
    #     unique_tags = set( [tag for sub_tag in list(df.Tag.values) for tag in ast.literal_eval(sub_tag)] )
    #     unique_tags.add("[START]")
    #     unique_tags.add("[END]")
    #     unique_tags.add("[PAD]")
    #     return unique_tags

    def extract_info(self,sentence):
        
        sentence = tf.strings.split( sentence, sep ="\t")[2]
        return sentence

    
    def preprocess_out_data(self, out_sentence ):
        preprocessed_tags_out = self.output_vectorizer(out_sentence)
        # print( self.count )
        # self.count += 1
        return preprocessed_tags_out
    
    def preprocess_inp_data(self, in_sentence ):
        preprocessed_tags_in = self.input_vectorizer(in_sentence)
        # print( self.count )
        # self.count += 1
        return preprocessed_tags_in

    def prepare_batch(self, inp, out ):
        return ( inp, out[:, :-1] ), out[: ,1:]

    def make_batches(self, ds):
        return (
            ds
            .shuffle(self.BUFFER_SIZE)
            .map(self.prepare_batch, tf.data.experimental.AUTOTUNE)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )

    def get_data_generators(self):
        
        inp_test_df = self.df.take(100)
        out_test_df = self.out_df.take(100)
        inp_train_df = self.df.skip(100)
        out_train_df = self.out_df.skip(100)

        train_generator  =   tf.data.Dataset.zip(inp_train_df.map(self.preprocess_inp_data), out_train_df.map( self.preprocess_out_data)) 
        test_generator  = tf.data.Dataset.zip(inp_test_df.map(self.preprocess_inp_data), out_test_df.map( self.preprocess_out_data) )

        train_batches = self.make_batches(train_generator)
        test_batches = self.make_batches(test_generator)

        return train_batches, test_batches


    def get_input_vocab_size(self):
        return self.input_vectorizer.vocabulary_size()
    
    def get_output_vocab_size(self):
        return self.output_vectorizer.vocabulary_size()
    
    def decode_input(self, vector):
        vocab = self.input_vectorizer.get_vocabulary()
        return( " ".join([vocab[each] for each in tf.squeeze(vector)]) )
    
    def decode_output(self, vector):
        vocab = self.output_vectorizer.get_vocabulary()
        return( " ".join([vocab[each] for each in tf.squeeze(vector)]) )
    
    def vectorize(self, sentences):
        return self.input_vectorizer(sentences)
    
    def encode_label(self, lst):
        return self.output_vectorizer(lst)
    
    # def inverse_transform_label(self, lst):
    #     return self.le.inverse_transform(lst)

