#!/user/bin/env python
# coding=utf-8
"""
@file: utils.py
@author: zwt
@time: 2020-10-31 11:35
@desc: 
"""
import torch
from collections import Counter
from torch.utils.data import Dataset
import os


# Define Dataset class
class Dataset_(Dataset):

    def __init__(self, args):
        self.args = args
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()

        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def findAllFile(self, base):
        data = []
        for root, ds, fs in os.walk(base):
            for f in fs:
                data.append(f)
        return data

    def load_words(self):
        # train_df = pd.read_csv('/Users/zhangweitao/Downloads/zwt/torch_practice/data/jokes.csv')
        # text = train_df['Joke'].str.cat(sep=' ')
        # return text.split(' ')
        text = []
        with open('../data/jay.txt', 'r', encoding='utf8') as f:
            for line in f.readlines():
                text.extend(line.strip().split())
        return text

    def get_unique_words(self):
        word_counts = Counter(self.words)

        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (torch.tensor(self.words_indexes[index:index + self.args.sequence_length]),
                torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]))
