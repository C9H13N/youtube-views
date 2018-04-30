import tensorflow as tf
import pvdm_params as params
import random
import traceback
import numpy as np
import re
import os
import pickle
import math

class TrainData:
    def __init__(self, paragraph_index=None):
        self.index = 0
        if paragraph_index is None:
            self.entries_ids = {}
        else:
            self.entries_ids = paragraph_index
        self.total_coint_unk = 0
        self.total_count_ok = 0
    def refine(row):
        row = row.replace('\n','').replace('\r', '').replace("\\n", "").lower()
        row = re.sub(r'((https|http)?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', row, flags=re.MULTILINE)
        remove_chars = ['(', ')', '[', ']', '-' ]
        chars = [".", "?", ',', '!', ':', ';']
        for char in remove_chars:
            row = row.replace(char, "")
        row = row.replace("  ", " ")
        for char in chars:
            row = row.replace(char, " "+char+" ")
        row = row.replace("  ", " ")

        return row
    def process_data(self, data, calc_words_ids = False):
        self.data = []
        append = False
        if len(self.entries_ids) < 1:
            append = True
        for entry in data:
            entry = TrainData.refine(entry)
            if append and entry not in self.entries_ids:
                self.entries_ids[entry] = len(self.entries_ids)
            if entry not in self.entries_ids:
                continue
            paragraph_id = self.entries_ids[entry]
            entry_words = entry.split()
            entry_words = entry_words + [params.padding_word]*max(0,params.word_windows_size - len(entry_words) + 1)
            for i in range(len(entry_words) - params.word_windows_size):
                self.data.append([paragraph_id, entry_words[i:i+params.word_windows_size], entry_words[i+params.word_windows_size]])
                # print("Paragraph: ",self.data[-1][0])
                # print("Context: ", self.data[-1][1])
                # print("Target: ", self.data[-1][2])
                # print("-"*10)
    def calc_words_ids(self):
        words_ids = {}
        inverse_words_ids = {}
        for entry in self.data:
            words = entry[1] + [entry[2]]
            for word in words:
                if word not in words_ids:
                    words_ids[word] = len(words_ids)
                    inverse_words_ids[words_ids[word]] = word
        if params.padding_word not in words_ids:
            words_ids[params.padding_word] = len(words_ids)
            inverse_words_ids[words_ids[params.padding_word]] = params.padding_word
        if params.unknown_token not in words_ids:
            words_ids[params.unknown_token] = len(words_ids)
            inverse_words_ids[words_ids[params.unknown_token]] = params.unknown_token
        return words_ids, inverse_words_ids
    def data_to_ids(self, words_dict):
        self.words_dict = words_dict
        for entry in self.data:
            entry[1] = [ words_dict[word] if word in words_dict else words_dict[params.unknown_token] for word in entry[1] ]
            if entry[2] in words_dict:
                entry[2] = words_dict[entry[2]]
                self.total_count_ok += 1
            else:
                entry[2] = words_dict[params.unknown_token]
                self.total_coint_unk += 1
    def get_next_batch(self, batch_size):
        if self.index + batch_size >= len(self.data):
            self.index = 0
        if self.index == 0:
            random.shuffle(self.data)
        result = self.data[self.index:self.index+batch_size]
        self.index += batch_size
        paragraph_ids, input_words, labels = zip(*result)
        paragraph_ids = np.asarray(paragraph_ids, dtype=np.int32)
        input_words = np.asarray(input_words, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        return paragraph_ids, input_words,labels
    def save_dicts(self, logdir):
        words_ids_filename = os.path.join(logdir, params.words_ids_filename)
        paragraphs_ids_filename = os.path.join(logdir, params.paragraphs_ids_filename)

        with open(words_ids_filename, 'wb') as pfile:
            pickle.dump(self.words_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paragraphs_ids_filename, 'wb') as pfile:
            pickle.dump(self.entries_ids, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        print("Dicts saved")
    def load_dicts(logdir):
        words_ids_filename = os.path.join(logdir, params.words_ids_filename)
        paragraphs_ids_filename = os.path.join(logdir, params.paragraphs_ids_filename)

        with open(words_ids_filename, "rb") as input_file:
            words_ids = pickle.load(input_file)
        with open(paragraphs_ids_filename, "rb") as input_file:
            paragraphs_ids = pickle.load(input_file)
        return words_ids, paragraphs_ids



class PVDM:
    def __init__(self, is_training, words_embeddings=None, paragraphs_embeddings=None, num_paragraphs=None, num_words=None, logdir="logdir"):
        self.is_training = is_training
        self.set_words_embeddings(words_embeddings, num_words)
        self.set_paragraph_embeddings(paragraphs_embeddings, num_paragraphs)
        self.logdir = logdir
    def create_model(self, words_count):
        self.words = tf.placeholder(dtype=tf.int32, shape=[None, params.word_windows_size])
        self.target_words = tf.placeholder(dtype=tf.int32, shape=[None])
        self.paragraph_id = tf.placeholder(dtype=tf.int32, shape=[None])

        self.words_embedding = self.get_words_embedding()
        self.paragraph_embedding = self.get_paragraph_embedding()
        self.target = tf.one_hot(self.target_words, words_count)

        self.words_embedded = tf.nn.embedding_lookup(self.words_embedding, self.words)
        self.paragraph_embedded = tf.nn.embedding_lookup(self.paragraph_embedding, self.paragraph_id)

        self.feature = tf.reduce_sum(self.words_embedded, axis=1)
        self.feature = (self.feature + self.paragraph_embedded)/(params.word_windows_size + 1)

        self.u_weight = tf.get_variable("softmax_weight", shape=[params.word_embedding_size, words_count],trainable=self.is_training)
        self.bias = tf.get_variable("softmax_bias", shape=[words_count], trainable=self.is_training)

        self.logits = tf.matmul(self.feature,self.u_weight) + self.bias

        self.prediction = tf.nn.softmax(self.logits, dim=-1)

        self.eps = 1e-4
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.target*tf.log(self.prediction+self.eps), axis=-1))
        # self.opt = tf.train.AdamOptimizer(params.learning_rate).minimize(self.loss)
        gs = tf.Variable(0, trainable=False)
        # self.opt = tf.train.AdagradOptimizer(params.learning_rate).minimize(self.loss,global_step=gs)
        if self.is_training:
            self.opt = tf.train.AdamOptimizer(params.learning_rate).minimize(self.loss,global_step=gs)
        else:
            self.opt = tf.train.GradientDescentOptimizer(params.learning_rate).minimize(self.loss,global_step=gs)
        self.prediction_ids = tf.cast(tf.argmax(self.prediction, axis=-1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction_ids, self.target_words), tf.float32))
        # self.opt = tf.train.GradientDescentOptimizer(params.learning_rate).minimize(self.loss,global_step=gs)

    def load_word_embedding(embeddings_filename):
        embeddings = []
        words_ids = {}
        inverse_word_ids = {}
        with open(embeddings_filename) as emb_file:
            for line in emb_file:

                entries = line.strip().split()
                word = entries[0]
                embedding = list(map(float,entries[-params.word_embedding_size:]))
                if not word in words_ids:
                    words_ids[word] = len(words_ids)
                    inverse_word_ids[words_ids[word]] = word
                    embeddings.append(embedding)

            if params.padding_word not in words_ids:
                words_ids[params.padding_word] = len(words_ids)
                inverse_word_ids[words_ids[params.padding_word]] = params.padding_word
                embedding = np.random.normal(0, .1, params.word_embedding_size)
                embeddings.append(embedding)
            if params.unknown_token not in words_ids:
                words_ids[params.unknown_token] = len(words_ids)
                inverse_word_ids[words_ids[params.unknown_token]] = params.unknown_token
                embedding = np.random.normal(0, .1, params.word_embedding_size)
                embeddings.append(embedding)
        return words_ids, inverse_word_ids, embeddings

    def set_words_embeddings(self, init_value=None, num_elements=None):
        if init_value is not None:
            init = tf.constant(init_value, dtype=tf.float32)
            self.word_embeddings_weights = tf.get_variable("words_embeddings_vectors", initializer=init, trainable=self.is_training)
        else:
            self.word_embeddings_weights = tf.get_variable("words_embeddings_vectors", shape=(num_elements, params.word_embedding_size), trainable=self.is_training)
    def set_paragraph_embeddings(self, init_value=None, num_elements=None):
        if init_value is not None:
            init = tf.constant(init_value, dtype=tf.float32)
            self.paragraph_embeddings_weights = tf.get_variable("paragraphs_embeddings_vectors", initializer=init)
        else:
            self.paragraph_embeddings_weights = tf.get_variable("paragraphs_embeddings_vectors", shape=[num_elements, params.word_embedding_size])
    def get_words_embedding(self):
        return self.word_embeddings_weights

    def get_paragraph_embedding(self):
        return self.paragraph_embeddings_weights

class Embedding:
    def __init__(self, feature_name):

        self.words_ids, self.paragraphs_ids  = TrainData.load_dicts(feature_name)
        self.graph = tf.Graph()
        self.session = tf.Session()
        self.words_counts = len(self.words_ids)

        self.feature_name = feature_name
        # with self.session as sess:

        self.load_graph()
    def load_graph(self, paragraps_emb=None, num_paragraphs=None):
        if num_paragraphs is None:
            num_paragraphs=len(self.paragraphs_ids)
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.session = tf.Session()
            self.model = PVDM(False, logdir=self.feature_name, num_words=self.words_counts,
                              num_paragraphs=num_paragraphs, paragraphs_embeddings=paragraps_emb)
            self.model.create_model(self.words_counts)
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, os.path.join(self.feature_name, params.best_test_logdir))
    def append_paragraph_embedding(self):
        with self.graph.as_default():
            # with self.session as sess:
            # print("Assign new paragraph vector")
            embedding = self.session.run(self.model.paragraph_embeddings_weights)
            glorot_limit = math.sqrt(6)/(math.sqrt(2*params.word_embedding_size))
            new_paragraph_vector = np.random.uniform(-glorot_limit, glorot_limit, size=(1, params.word_embedding_size))
            new_embedding = np.concatenate([embedding, new_paragraph_vector], axis=0)
        self.session.close()
        tf.reset_default_graph()

        # self.load_graph(paragraps_emb=new_embedding)
        self.load_graph(num_paragraphs=len(self.paragraphs_ids)-1)
        with self.graph.as_default():
            self.session.run(tf.assign(self.model.paragraph_embeddings_weights,new_embedding,validate_shape=False))
    def train_new_paragraph(self, paragraph, paragraph_orig):
        data_master = TrainData(paragraph_index=self.paragraphs_ids )
        data_master.process_data([paragraph_orig])
        data_master.data_to_ids(self.words_ids)
        self.append_paragraph_embedding()
        with self.graph.as_default():
            prev_loss = 1e6
            loss = 0

            # with self.session as sess:


            iteration_number = params.min_iteration_number
            print('-'*10,"Start learning new paragraph", '-'*10)
            while math.fabs(loss - prev_loss) > params.eps_learn_stop or iteration_number > 0:
                if iteration_number < params.min_iteration_number - params.max_iterations_number:
                    break
                prev_loss = loss
                paragraphs_ids, words_ids, labels_ids = data_master.get_next_batch(params.batch_size)
                accuracy, loss, _ = self.session.run([self.model.accuracy, self.model.loss, self.model.opt],
                                               feed_dict={self.model.paragraph_id: paragraphs_ids,
                                                          self.model.words: words_ids, self.model.target_words: labels_ids})
                iteration_number -= 1
                print("Learning paragraph loss: %f | Previous loss: %f | Accuracy: %f "%(loss,prev_loss,accuracy))

            if accuracy < 0.2:
                print(paragraph)
            # print("Trained new paragraph")
            self.saver.save(self.session, os.path.join(self.feature_name, params.best_test_logdir))
            data_master.save_dicts(self.feature_name)

            print('-' * 10, "----------------------------", '-' * 10)
            print()
    def get_embedding(self, paragraph):
        paragraph_id = self.paragraphs_ids[paragraph]
        with self.graph.as_default():
            # with self.session as sess:
            embedding = self.session.run(tf.nn.embedding_lookup(self.model.paragraph_embeddings_weights, paragraph_id))
        return embedding
    def get_embedding_multiple(self, paragraphs):
        paragraphs_ids = [self.paragraphs_ids[paragraph] for paragraph in paragraphs]
        with self.graph.as_default():
            # with self.session as sess:
            embeddings = self.session.run(tf.nn.embedding_lookup(self.model.paragraph_embeddings_weights, paragraphs_ids))
        return embeddings
    def vectorize(self, paragraph):
        paragraph_orig = paragraph
        paragraph = TrainData.refine(paragraph)
        if paragraph not in self.paragraphs_ids:
            # print("Size before: ", len(self.paragraphs_ids))
            self.paragraphs_ids[paragraph] = len(self.paragraphs_ids)
            # print("Size after: ", len(self.paragraphs_ids))
            self.train_new_paragraph(paragraph, paragraph_orig)
        return self.get_embedding(paragraph)
    def vectorize_paragraphs(self, paragraphs):
        paragraphs_orig = paragraphs
        paragraphs = [TrainData.refine(paragraph) for paragraph in paragraphs]
        for paragraph_index in range(len(paragraphs)):
            if paragraphs[paragraph_index] not in self.paragraphs_ids:
                # print("Size before: ", len(self.paragraphs_ids))
                paragraph = paragraphs[paragraph_index]
                paragraph_orig = paragraphs_orig[paragraph_index]

                self.paragraphs_ids[paragraph] = len(self.paragraphs_ids)
                # print("Size after: ", len(self.paragraphs_ids))
                self.train_new_paragraph(paragraph, paragraph_orig)
        return self.get_embedding_multiple(paragraphs)
        # return self.get_embedding(paragraph)


