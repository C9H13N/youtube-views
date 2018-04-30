from train_baseline import one_hot_feature_pre, load_multiple, log_label, baseline_category_and_chanel, baseline_category_id,baseline_chanel_label,baseline_tfidf, onehot
import train_prediction_with_pvdm as pvdm_models
import pvdm_embedding as emb_model
import numpy as np
from scipy import sparse
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def get_one_hot_vectors(chanels, categories):
    onehot_embedding_channels, channels_ids = one_hot_feature_pre(chanels)
    onehot_embedding_categories, categories_ids = one_hot_feature_pre(categories)
    return onehot_embedding_channels, channels_ids, onehot_embedding_categories, categories_ids


def test_model(model, inputs):
    return model.predict(inputs)

def tf_idf_inputs(descriptions, titles, channels_onehot, categories_onehot, title_vect, descr_vect):
    titles = title_vect.transform(titles)
    descriptions = descr_vect.transform(descriptions)

    channels_onehot = np.asarray(channels_onehot, dtype=np.float32)
    categories_onehot = np.asarray(categories_onehot, dtype=np.float32)
    channels_onehot = sparse.csr_matrix(channels_onehot)
    categories_onehot = sparse.csr_matrix(categories_onehot)

    return sparse.hstack([channels_onehot, categories_onehot, titles, descriptions], format='csr')
def pvdm_inputs(descriptions, titles,channels_onehot,categories_onehot, emb_descr, emb_title):
    descr_emb = emb_descr.vectorize_paragraphs(descriptions)
    # title_emb = [emb_encoder_title.vectorize(feature) for feature in titles]
    title_emb = emb_title.vectorize_paragraphs(titles)
    channels_onehot = np.asarray(channels_onehot, dtype=np.float32)
    categories_onehot = np.asarray(categories_onehot, dtype=np.float32)
    input = [np.concatenate((channels_onehot[i], categories_onehot[i], descr_emb[i], title_emb[i])) for i in
             range(len(channels_onehot))]
    return input


def eval():
    train_data = "train.csv"
    valid_data = "valid.csv"

    descriptions, titles, chanels, categories, views, likes, dislikes = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float])



    emb_encoder_descr = emb_model.Embedding("description")
    emb_encoder_title = emb_model.Embedding("title")



    models = {}

    model_view, model_like, model_dislike = baseline_category_and_chanel()
    models["category_channel_onehot"] = (model_view, model_like, model_dislike)
    model_view, model_like, model_dislike = baseline_category_id()
    models["category_onehot"] = (model_view, model_like, model_dislike)
    model_view, model_like, model_dislike = baseline_chanel_label()
    models["channel_onehot"] = (model_view, model_like, model_dislike)
    model_view, model_like, model_dislike,vectorizer_title, vectorizer_descr = baseline_tfidf()
    models["tfidf"] = (model_view, model_like, model_dislike)
    model_view, model_like, model_dislike = pvdm_models.load_model()
    models["pvdm"] = (model_view, model_like, model_dislike)

    descriptions_valid, titles_valid, chanels_valid, categories_valid, views_valid, likes_valid, dislikes_valid = \
        load_multiple(["description", "title", "channel_title", "category_id", "views", "likes", "dislikes"],
                      [str, str, str, str, float, float, float], filename=valid_data)
    views_valid = log_label(views_valid)
    likes_valid = log_label(likes_valid)
    dislikes_valid = log_label(dislikes_valid)

    chanels_test, categories_test = \
        load_multiple(["channel_title", "category_id"],
                      [str, str], filename="test.csv")
    onehot_embedding_channels, channels_ids = one_hot_feature_pre(chanels+chanels_valid+chanels_test)
    onehot_embedding_categories, categories_ids = one_hot_feature_pre(categories+categories_valid+categories_test)

    channels_onehot = [onehot(channel, channels_ids, onehot_embedding_channels) for channel in chanels_valid]
    categories_onehot = [onehot(category, categories_ids, onehot_embedding_categories) for category in categories_valid]

    inputs = {}

    inputs["category_channel_onehot"] = [np.concatenate((channels_onehot[i], categories_onehot[i])) for i in range(len(channels_onehot))]
    inputs["category_onehot"] = categories_onehot
    inputs["channel_onehot"] = channels_onehot
    inputs["tfidf"] = tf_idf_inputs(descriptions_valid, titles_valid, channels_onehot, categories_onehot,vectorizer_title,vectorizer_descr)
    inputs["pvdm"] = pvdm_inputs(descriptions_valid, titles_valid, channels_onehot, categories_onehot,emb_encoder_descr, emb_encoder_title)

    labels = (views_valid, likes_valid, dislikes_valid)
    labels_names = ("views", "likes", "dislikes")
    scores = {"r2" : r2_score, "pearson":pearsonr}
    results = {}
    for model_name in models:
        if model_name not in results:
            results[model_name] = {}
        for i in range(len(models[model_name])):
            predicted = models[model_name][i].predict(inputs[model_name])
            gr_truth = labels[i]
            if labels_names[i] not in results[model_name]:
                results[model_name][labels_names[i]] = {score:0 for score in scores}
            for score in scores:
                score_value = scores[score](gr_truth, predicted)
                results[model_name][labels_names[i]][score] = score_value
    return results

if __name__ == "__main__":
    print(eval())
