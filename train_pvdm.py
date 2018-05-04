import pvdm_embedding as emb_model
import pvdm_params as params
import tensorflow as tf
import data as datahelper
import numpy as np
import os
import sys

def load_data(data_filename, feature_name):
    data = datahelper.read_data(data_filename)
    data = data.replace(np.nan, '', regex=True)
    data = data[feature_name]

    return data.tolist()

def train(data_filename,test_filename, feature_name):
    graph = tf.Graph()
    if not os.path.exists(feature_name):
        os.mkdir(feature_name)
    all_data = load_data(data_filename, feature_name)
    data_master = emb_model.TrainData()
    data_master.process_data(all_data)

    test_data_master = emb_model.TrainData(paragraph_index=data_master.entries_ids)
    test_data = load_data(test_filename, feature_name)
    test_data_master.process_data(test_data)


    dataset_size =  len(data_master.data)


    if not params.use_pretrained_embeddings:
        words_ids, inverse_word_ids = data_master.calc_words_ids()
    else:
        print("Using pretrained embeddings")
        words_ids, inverse_word_ids, embeddings = emb_model.PVDM.load_word_embedding(params.embedding_filename)
    data_master.data_to_ids(words_ids)
    test_data_master.data_to_ids(words_ids)

    print("Total unk: ", data_master.total_coint_unk)
    print("Total others: ", data_master.total_count_ok)
    # embeddings = np.asarray(embeddings, dtype=np.float32)

    #Saving words and paragraphs dicts
    data_master.save_dicts(feature_name)

    with graph.as_default():
        # model = emb_model.PVDM(is_training=True,words_embeddings=embeddings, num_paragraphs=len(data_master.entries_ids))
        model = emb_model.PVDM(is_training=True,num_words = len(words_ids), num_paragraphs=len(data_master.entries_ids), logdir=feature_name)
        model.create_model(words_count=len(words_ids))
    best_accuracy = 0
    supervisor = tf.train.Supervisor(graph)
    with supervisor.managed_session() as sess:
        for epoch in range(100):
            for i in range(dataset_size//params.batch_size):
                paragraphs_ids, words_ids, labels_ids = data_master.get_next_batch(params.batch_size)
                accuracy, pred, loss, _ = sess.run([model.accuracy, model.prediction, model.loss, model.opt], feed_dict={model.paragraph_id:paragraphs_ids, model.words:words_ids, model.target_words:labels_ids})
                paragraphs_ids_test, words_ids_test, labels_ids_test = test_data_master.get_next_batch(params.test_batch_size)
                accuracy_test = sess.run(model.accuracy, feed_dict={model.paragraph_id:paragraphs_ids_test, model.words:words_ids_test, model.target_words:labels_ids_test})
                # print("Loss : ", loss)
                # print("Accuracy: ", accuracy)
                # print("Accuracy test: ", accuracy_test)
                if accuracy_test > best_accuracy:
                    best_accuracy = accuracy_test
                    print("Best accuracy reached : %f | On epoch: %d "%(best_accuracy,epoch))
                    supervisor.saver.save(sess, os.path.join(feature_name, params.best_test_logdir))
                # print("-"*10)
                pred = np.argmax(pred, axis=-1)
        words_predicted = []
        for i in range(pred.shape[0]):
            words_predicted.append(inverse_word_ids[pred[i]])
    print("Predicted words: ", "\n".join(words_predicted))




def eval(filename, feature_name):
    all_data = load_data(filename, feature_name)
    emb_encoder = emb_model.Embedding(feature_name)
    for feature in all_data:
        res = emb_encoder.vectorize(feature)




if __name__ == "__main__":

    # words_ids, inverse_words_ids, embeddings =  emb_model.PVDM.load_word_embedding(params.embedding_filename)
    # print("Len words: ", len(words_ids))
    dataset_filename = "train.csv"
    valid_dataset_filename = "valid.csv"
    test_dataset_filename = "test.csv"
    if len(sys.argv) > 1:
        feature_name = sys.argv[1]
    else:
        feature_name = "title"
        # feature_name = "description"

    # eval(valid_dataset_filename, feature_name)
    # feature_name = "title"
    train(data_filename=dataset_filename, test_filename=test_dataset_filename, feature_name=feature_name)