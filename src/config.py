import os
import json


def get_config(version):
    CONFIG_PATH = 'D:\\Code\\Kaggle\\Movie Review Sentiment Analysis\\configs'
    f = open(os.path.join(CONFIG_PATH, '{}.json'.format(version)), 'r')
    config = json.load(f)
    f.close()
    return config