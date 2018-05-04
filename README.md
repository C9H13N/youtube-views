# youtube-views
Youtube videos likes, dislikes and views prediction from video metadata
# Requirements
Project tested on ubuntu 16 lts, python version 3.5, tensorflow version 1.3
* Pandas
* Tensorflow
* matplotlib
* langdetect
* numpy
* re
* scipy
* scikit-learn
# Installation
* Download project
* Additional download pvdm model checkpoints from https://drive.google.com/open?id=1ox6Br02AZ2NaIR3BZg9AUrfPO64jouDP (or train yourself)
* Unzip checkpoints archive in project folder
# Results
Results are in results.ipynb note. For each model shown results for each target (views, likes, dislikes). For evaluation metric i use r2 score and pearson correlation( in result 2 values, first is correlation coefficient, second is p value)
# Usage
* For dataset splitting run data.py ( used only US csv, pandas can't read correctly other files)
* For training pvdm model run train_pvdm.py <feature_name> (feature name is column name in dataset, example: python3 train_pvdm.py description)
    For word embeddings you can use pretrained vectors. Change use_pretrained_embeddings value in config file (pvdm_params.py) to True and vectors from embedding_filename will be used.
* For results run evaluation.py
# TODO
* Fix pandas issues while reading csv files except US

