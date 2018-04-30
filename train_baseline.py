import data
import numpy as np
import math
from sklearn.linear_model import Ridge
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

train_dataname = "train.csv"

def load_data(feature_key, target_key, dtypes=[str,str]):
    df = data.read_data(train_dataname)
    df = df.dropna(subset=["views", "likes", "title", "channel_title", "description", "dislikes", "category_id"])

    features = df[feature_key]
    labels = df[target_key]

    features = list(map(dtypes[0], features))
    labels = list(map(dtypes[1], labels))
    return features, labels
def load_multiple(columns, dtypes, filename="train.csv"):
    df = data.read_data(filename)
    df = df.dropna(subset=["views", "likes", "title", "channel_title", "description", "dislikes", "category_id"])
    columns = [df[key] for key in columns]

    columns = [list(map(dtypes[i], columns[i])) for i in range(len(columns))]
    return columns

def one_hot_feature_pre(features):
    features_id = {}
    for feature in features:
        if feature not in features_id:
            features_id[feature] = len(features_id)
    onehot_embedding = np.eye(len(features_id))
    return onehot_embedding, features_id
def onehot(feature, features_id, onehot_embedding):
    return onehot_embedding[features_id[feature]]
def log_label(labels):
    for label in labels:
        log_res = math.log(label + 1)
        if math.isnan(log_res):
            print("Bad label: ", label)
    labels = [math.log(label+1) for label in labels]
    return labels
def ridge_model(features, labels):
    clf = Ridge(alpha=1.0)
    clf.fit(features, labels)
    return clf

def tfidf_data(data):
    vectorizer = TfidfVectorizer()
    res = vectorizer.fit_transform(data)
    return res, vectorizer
def evaluate_model(model, features):
    return model.predict(features)
def baseline_category_and_chanel():
    descriptions, titles, chanels, categories, views, likes, dislikes = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float])
    views = log_label(views)
    likes = log_label(likes)
    dislikes = log_label(dislikes)

    chanels_test, categories_test = \
        load_multiple(["channel_title", "category_id"],
                      [str, str], filename="test.csv")
    chanels_valid, categories_valid = \
        load_multiple(["channel_title", "category_id"],
                      [str, str], filename="valid.csv")

    onehot_embedding_channels, channels_ids = one_hot_feature_pre(chanels+chanels_valid+chanels_test)
    onehot_embedding_categories, categories_ids = one_hot_feature_pre(categories+categories_valid+categories_test)
    channels_onehot = [onehot(channel, channels_ids, onehot_embedding_channels) for channel in chanels]
    categories_onehot = [onehot(category, categories_ids, onehot_embedding_categories) for category in categories]

    input = [np.concatenate( (channels_onehot[i],categories_onehot[i])) for i in range(len(channels_onehot))]
    model_views = ridge_model(input, views)
    model_likes = ridge_model(input, likes)
    model_dislikes = ridge_model(input, dislikes)
    return model_views, model_likes, model_dislikes
def baseline_category_id():
    descriptions, titles, chanels, categories, views, likes, dislikes = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float])

    views = log_label(views)
    likes = log_label(likes)
    dislikes = log_label(dislikes)
    categories_test = \
        load_multiple(["category_id"],
                      [str], filename="test.csv")[0]
    categories_valid = \
        load_multiple(["category_id"],
                      [str], filename="valid.csv")[0]
    onehot_embedding, channels_ids = one_hot_feature_pre(categories + categories_test + categories_valid)
    channels_onehot = [onehot(channel, channels_ids, onehot_embedding) for channel in categories]

    model_views = ridge_model(channels_onehot, views)
    model_likes = ridge_model(channels_onehot, likes)
    model_dislikes = ridge_model(channels_onehot, dislikes)
    return model_views, model_likes, model_dislikes

def baseline_chanel_label():
    descriptions, titles, chanels, categories, views, likes, dislikes = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float])
    views = log_label(views)
    likes = log_label(likes)
    dislikes = log_label(dislikes)

    chanels_test = \
        load_multiple(["channel_title"],
                      [str], filename="test.csv")[0]
    chanels_valid = \
        load_multiple(["channel_title"],
                      [str], filename="valid.csv")[0]

    onehot_embedding, channels_ids = one_hot_feature_pre(chanels+chanels_test+chanels_valid)
    channels_onehot = [onehot(channel,channels_ids, onehot_embedding) for channel in chanels]

    model_views = ridge_model(channels_onehot, views)
    model_likes = ridge_model(channels_onehot, likes)
    model_dislikes = ridge_model(channels_onehot, dislikes)
    return model_views, model_likes, model_dislikes

def baseline_tfidf():

    descriptions, titles, chanels, categories, views, likes, dislikes = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float])

    chanels_test, categories_test = \
        load_multiple(["channel_title", "category_id"],
                      [str, str], filename="test.csv")
    chanels_valid, categories_valid = \
        load_multiple(["channel_title", "category_id"],
                      [str, str], filename="valid.csv")

    onehot_embedding_channels, channels_ids = one_hot_feature_pre(chanels+chanels_valid+chanels_test)
    onehot_embedding_categories, categories_ids = one_hot_feature_pre(categories+categories_valid+categories_test)
    channels_onehot = [onehot(channel, channels_ids, onehot_embedding_channels) for channel in chanels]
    categories_onehot = [onehot(category, categories_ids, onehot_embedding_categories) for category in categories]

    views = log_label(views)
    likes = log_label(likes)
    dislikes = log_label(dislikes)

    titles,vectorizer_title = tfidf_data(titles)
    descriptions, vectorizer_descr = tfidf_data(descriptions)

    channels_onehot = np.asarray(channels_onehot, dtype=np.float32)
    categories_onehot = np.asarray(categories_onehot, dtype=np.float32)
    channels_onehot = sparse.csr_matrix(channels_onehot)
    categories_onehot = sparse.csr_matrix(categories_onehot)

    input = sparse.hstack([channels_onehot, categories_onehot, titles, descriptions], format='csr')
    model_views = ridge_model(input, views)
    model_likes = ridge_model(input, likes)
    model_dislikes = ridge_model(input, dislikes)

    return model_views, model_likes, model_dislikes, vectorizer_title, vectorizer_descr



if __name__ == "__main__":
    # baseline_chanel_label()
    # baseline_category_id()
    # baseline_category_and_chanel()
    baseline_tfidf()