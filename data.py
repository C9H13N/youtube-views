import pandas as pd
from dateutil import parser
import datetime
import pytz
# from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback
import csv
from langdetect import detect
# now = datetime.datetime.now(datetime.timezone.utc)
import math
days_interpolation = 6

def tags_parser(tag):
    # tags = tag.split("|")
    # tags = [t.replace('"', '') for t in tags]
    tags = tag.replace('"', '')
    return tags
def filter_data(dataframe):
    # keep_keys = ["video_id", "trending_date", "title", ]
    print("Dataframe shape: {}".format(dataframe.shape))
    keys = ["views", "likes", "dislikes", "comment_count"]
    for i, row in dataframe.iterrows():
        # print(type(row["title"]))
        print("Row {} processing starting".format(i))
        try:
            if detect(str(row["title"])) != "en" or detect(str(row["description"])) != "en":
                dataframe.drop(dataframe.index[i])
            publish_time = parser.parse(row["publish_time"])
            trending_time = datetime.datetime.strptime(row["trending_date"], "%y.%d.%m")
            trending_time = pytz.utc.localize(trending_time)
            delta = trending_time - publish_time
            delta = delta.total_seconds()+1
            # delta /= 60 * 60 * 24
            delta /= 60
            delta = math.log(delta+1)
            interpolation_seconds = days_interpolation*60*24#24*60*60
            interpolation_seconds = math.log(interpolation_seconds)
            scale_factor = interpolation_seconds/delta
            for key in keys:
                dataframe.set_value(i, key, float(row[key])*scale_factor)
            dataframe.set_value(i, "tags", tags_parser(row["tags"]))
        except:
            print(row["title"], " | ", row["description"])
            traceback.print_exc()
            dataframe.drop(dataframe.index[i])

    return dataframe
def shuffle_dataframe(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def calc_deltas_frequency_stat(dataframe):
    delta_stats = {}
    for i, row in dataframe.iterrows():
        key = "tags"
        # dataframe.set_value(i, key, tags_parser(dataframe.iloc[i][key]))
        try:
            publish_time = parser.parse(row["publish_time"])
            trending_time = datetime.datetime.strptime(row["trending_date"], "%y.%d.%m")
            trending_time = pytz.utc.localize(trending_time)
            # trending_time = parser.parse(dataframe.iloc[0]["trending_date"])
            delta = trending_time - publish_time
            delta = delta.total_seconds()
            delta /= 60 * 60 * 24
            delta = int(delta)
            # print(delta)
            if delta not in delta_stats:
                delta_stats[delta] = 0
            delta_stats[delta] += 1
        except:
            print("ERROR ROW: ",row)
            traceback.print_exc()
    return delta_stats
def draw_plot_deltas_distribution(delta_stats):
    deltas = list(delta_stats.items())
    deltas, counts = zip(*deltas)
    plt.plot(deltas, counts, 'ro')

    plt.axis([-1, 30, 0, max(counts)])
    plt.xticks(np.arange(-1, 30, 1))
    plt.show()

def read_data(data_filename):
    data = pd.read_csv(data_filename, quotechar='"', quoting=csv.QUOTE_ALL)
    return data
    # keys = ["video_id", "trending_date", "title", "channel_title", "category_id", "publish_time", "tags", "views", "likes", "dislikes", "comment_count", "description"]
    # data = list(zip(*[list(data[key]) for key in keys]))
    # print(data[0])

def prepare_all_data(root_folder):
    datas = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".csv":
                datas.append(read_data(os.path.join(root, file)))
                print("Reading {} file".format(file))
            else:
                print("Skip {} file".format(file))

    return pd.concat(datas)

def split_dataframe(dataframe, proportions = [0.6, 0.2, 0.2]):
    train = dataframe.sample(frac=proportions[0])
    remain = dataframe.drop(train.index)
    proportions[1] = proportions[1]/(proportions[1]+proportions[2])
    proportions[2] = 1 - proportions[1]
    test = remain.sample(frac=proportions[1])
    valid = remain.drop(test.index)
    return train, test, valid

def save_dataframe(data, filename):
    data.to_csv(filename, sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    root_data_folder = "./youtube-new"
    data = read_data("./youtube-new/USvideos.csv")
    # data = prepare_all_data(root_data_folder)
    # stats = calc_deltas_frequency_stat(data)
    # delta_stats = filter_data(data)
    # draw_plot_deltas_distribution(stats)
    data = filter_data(data)
    data = shuffle_dataframe(data)
    train, test, valid = split_dataframe(data)
    save_dataframe(train, "train.csv")
    save_dataframe(test, "test.csv")
    save_dataframe(valid, "valid.csv")
    # save_dataframe(data,"dataGB.csv")
