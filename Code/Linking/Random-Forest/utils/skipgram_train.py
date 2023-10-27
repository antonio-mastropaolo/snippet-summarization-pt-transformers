from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api

import os
import csv
import sys
import re
import  string
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd

import random
random.seed(43)

ps = PorterStemmer()

nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))


class SkipGram():
    def __init__(self, input_path):
        self.input_path=input_path

        self.input_file = os.path.join(input_path, "train.csv")

    def read_data(self, data_path):
        file = open(data_path, 'r')
        csvreader = csv.reader(file)

        rows = []
        for row in csvreader:
            rows.append(row)

        file.close()

        print("Read {} lines".format(len(rows)))
        return rows

    def process_text(self, text):
        '''
        process the text (the comment or the statement code)
        '''
        # comment="the programmer programmed a new file"

        text_new=text

        for character in string.punctuation:
            text_new = text_new.replace(character, ' ')

        tokens = nltk.word_tokenize(text_new)

        text = " ".join(tokens)

        regex = "(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"

        tokens = re.split(regex, text)

        res = list()
        for t in tokens:
            # we remove the punctuation. Not said in the paper but they use "words"
            t_currs = t.split(" ")

            for t_curr in t_currs:

                # for character in string.punctuation:
                #     t_curr = t_curr.replace(character, ' ')

                if len(t_curr) > 0:
                    res.append(t_curr.lower())

        filtered_tokens = [w for w in res if not w in stop_words]

        res = list()
        for w in filtered_tokens:
            res.append(ps.stem(w))

        filtered_tokens = res

        return filtered_tokens

    def sample_two_random_tokens(self, current_token, list_for_sampling):
        if len(list_for_sampling)<2:
            return list()
        left=random.sample(list_for_sampling,2)
        right=random.sample(list_for_sampling,2)
        res=list()
        res.extend(left)
        res.append(current_token)
        res.extend(right)
        return res

    def clean_comment(self, comment):
        comment_list = eval(comment)
        comment_list = [c.replace("/*", "").replace("*/", "").replace("//", "").strip() for c in comment_list]
        return " ".join(comment_list)

    def create_dataset(self):
        # we create the dataset that the authors described in the paper (putting close some tokens from comment
        # and some tokens from the referenced lines
        # lines=self.read_data(self.input_file)

        lines=pd.read_csv(self.input_file)

        dataset_comment=list()
        dataset_code=list()

        for index, l in lines.iterrows():

            # print(l["comment"])


            comment = self.clean_comment(l["comment"])
            # print(comment)
            referenced_code=eval(l["documentedCode"])
            # print(referenced_code)
            referenced_code=" ".join(referenced_code)
            # print(comment)
            # print(referenced_code)
            comment_tokens=self.process_text(comment)
            # print(comment_tokens)
            referenced_tokens=self.process_text(referenced_code)
            # print(referenced_tokens)

            list_curr = list()

            for t in comment_tokens:
                curr=self.sample_two_random_tokens(t, referenced_tokens)
                if len(curr)>0:
                    list_curr.extend(curr)

            if len(list_curr)>0:
                dataset_comment.append(list_curr)

            list_curr=list()

            for t in referenced_tokens:
                curr=self.sample_two_random_tokens(t, comment_tokens)
                if len(curr)>0:
                    list_curr.extend(curr)

            if len(list_curr)>0:
                dataset_code.append(list_curr)

        # print(len(dataset_code))
        # print(len(dataset_comment))

        final_dataset=dataset_comment
        final_dataset.extend(dataset_code)
        # print(len(final_dataset))
        return final_dataset

    def train_skipgram_model(self):
        dataset=self.create_dataset()
        print("{} DATASET INSTANCES".format(len(dataset)))
        # Train Word2Vec model. Defaults result vector size = 100
        print("START TRAINING")
        model = Word2Vec(dataset, window=5, min_count = 0, workers=cpu_count())

        model.save("skipgram_model.model")

        print("MODEL TRAINED")
