from train_baseline import load_data, log_label, evaluate_model, one_hot_feature_pre, onehot, ridge_model, load_multiple
import math
import numpy as np
import pvdm_embedding as emb_model
import pvdm_params as params
import pickle




def pvdm_ridge():
    # descriptions, _ = load_data(feature_key="description", target_key="views", dtypes=[str, float])
    # titles, _ = load_data(feature_key="title", target_key="views", dtypes=[str, float])
    # chanels, _ = load_data(feature_key="channel_title", target_key="views", dtypes=[str, float])
    # categories, views = load_data(feature_key="category_id", target_key="views", dtypes=[str, float])
    # _, likes = load_data(feature_key="category_id", target_key="likes", dtypes=[str, float])
    # _, dislikes = load_data(feature_key="category_id", target_key="dislikes", dtypes=[str, float])

    descriptions, titles, chanels, categories, views,likes, dislikes = \
        load_multiple(["description","title","channel_title", "category_id", "views", "likes", "dislikes"],[str,str,str,str,float,float,float])
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
    # print("Channels shape: ", np.asarray(channels_onehot, dtype=np.float32).shape)
    # print("Labels shape: ", np.asarray(labels, dtype=np.float32).shape)
    emb_encoder_descr = emb_model.Embedding("description")
    emb_encoder_title = emb_model.Embedding("title")
    print("Start vectorising descriptions")

    descr_emb = emb_encoder_descr.vectorize_paragraphs(descriptions)
    # title_emb = [emb_encoder_title.vectorize(feature) for feature in titles]
    title_emb = emb_encoder_title.vectorize_paragraphs(titles)
    channels_onehot = np.asarray(channels_onehot, dtype=np.float32)
    categories_onehot = np.asarray(categories_onehot, dtype=np.float32)
    input = [np.concatenate((channels_onehot[i], categories_onehot[i], descr_emb[i], title_emb[i])) for i in range(len(channels_onehot))]
    input = np.asarray(input, dtype=np.float32)
    views = np.asarray(views, dtype=np.float32)
    model_views = ridge_model(input, views)
    model_likes = ridge_model(input, likes)
    model_dislikes = ridge_model(input, dislikes)
    return model_views, model_likes, model_dislikes

def dump_models(model_views, model_likes, model_dislikes):
    with open(params.model_dislikes_filename, 'wb') as pfile:
        pickle.dump(model_dislikes, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(params.model_likes_filename, 'wb') as pfile:
        pickle.dump(model_likes, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(params.model_views_filename, 'wb') as pfile:
        pickle.dump(model_views, pfile, protocol=pickle.HIGHEST_PROTOCOL)
def load_model():
    # with open(params.model_dislikes_filename, 'rb') as pfile:
    #     dislikes_model = pickle.load(pfile)
    # with open(params.model_likes_filename, 'rb') as pfile:
    #     likes_model = pickle.load(pfile)
    # with open(params.model_views_filename, 'rb') as pfile:
    #     views_model = pickle.load(pfile)
    # return views_model, likes_model, dislikes_model
    return pvdm_ridge()

if __name__ == "__main__":
    model_views, model_likes, model_dislikes = pvdm_ridge()
    dump_models(model_views, model_likes, model_dislikes)
