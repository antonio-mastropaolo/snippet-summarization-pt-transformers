import sys
import os
import csv
import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd

ps = PorterStemmer()


nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))


class CommentFeatures():
	def __init__(self, separator, set, input_folder):
		self.separator=separator
		self.header=["id_instance", "num tokens", "num nouns", "num verbs", "next line blank"]
		self.set=set

		self.input_folder = input_folder

		self.input_files = [f for f in os.listdir(input_folder) if ".csv" in f]



	def read_data(self, data_path):
		file = open(data_path, 'r')
		csvreader = csv.reader(file)

		rows = []
		for row in csvreader:
			rows.append(row)

		file.close()

		print("Read {} lines".format(len(rows)))
		return rows

	def process_comment(self, comment, with_stemming=True):

		# comment="the programmer programmed a new file"

		text_new = comment

		for character in string.punctuation:
			text_new = text_new.replace(character, ' ')

		tokens=nltk.word_tokenize(text_new)

		text=" ".join(tokens)

		regex="(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"

		tokens=re.split(regex, text)

		res=list()
		for t in tokens:
			# we remove the punctuation. Not said in the paper but they use "words"
			t_currs=t.split(" ")

			for t_curr in t_currs:

				if len(t_curr)>0:
					res.append(t_curr.lower())


		filtered_tokens = [w for w in res if not w in stop_words]


		if with_stemming:
			res=list()
			for w in filtered_tokens:
				res.append(ps.stem(w))

			filtered_tokens=res

		# print(comment)
		# print(filtered_tokens)

		pos_tagged = nltk.pos_tag(filtered_tokens)
		# print(pos_tagged)

		num_verbs=0
		num_nouns=0
		num_words=len(filtered_tokens)

		for p in pos_tagged:
			if p[1][:2]=="NN":
				num_nouns+=1
			elif p[1][:2]=="VB":
				num_verbs+=1

		return num_words, num_nouns, num_verbs

	def clean_comment(self, comment):
		comment_list = eval(comment)
		comment_list = [c.replace("/*", "").replace("*/", "").replace("//", "").strip() for c in comment_list]
		return " ".join(comment_list)

	def process_files(self):

		for file in self.input_files:
			if self.set not in file:
				continue

			lines=pd.read_csv(os.path.join(self.input_folder, file))
			errors = 0

			features_with_stemming = dict()
			features_without_stemming = dict()

			print("COMMENT FEATURES {} LINES".format(len(lines)))
			for index, line in lines.iterrows():
				try:

					id_instance = int(line["index"])
					print(id_instance)

					comment = self.clean_comment(line["comment"])

					code_start_end=line["linkingInstance"]


					feature_with_stemming, feature_without_stemming = self.compute_features(id_instance, comment, code_start_end)

					features_with_stemming[id_instance]=feature_with_stemming
					features_without_stemming[id_instance]=feature_without_stemming


				except Exception as e:
					errors += 1
					print("ERROR__ERROR")
					print(e)

			print("{} ERRORS".format(errors))

			return self.header, features_with_stemming, features_without_stemming

	def compute_features(self, id_instance, comment, code_start_end):

		# we check whether the line after the next comment is blank or not
		parts=code_start_end.split("</comment>")

		lines=parts[-1].split("\n")
		next_blank_line=0
		# print(lines[1])
		if len((lines[1].replace("<start>","").replace("<end>","")).strip())==0:
			next_blank_line=1



		num_tokens_with_stem, num_nouns_with_stem, num_verbs_with_stem=self.process_comment(comment, True )
		num_tokens_without_stem, num_nouns_without_stem, num_verbs_without_stem=self.process_comment(comment, False )

		values_with_stemming=list()
		values_with_stemming.append(str(id_instance))
		values_with_stemming.append(str(num_tokens_with_stem))
		values_with_stemming.append(str(num_nouns_with_stem))
		values_with_stemming.append(str(num_verbs_with_stem))
		values_with_stemming.append(str(next_blank_line))

		result_with_stemming=self.separator.join(values_with_stemming)

		values_without_stemming = list()
		values_without_stemming.append(str(id_instance))
		values_without_stemming.append(str(num_tokens_without_stem))
		values_without_stemming.append(str(num_nouns_without_stem))
		values_without_stemming.append(str(num_verbs_without_stem))
		values_without_stemming.append(str(next_blank_line))

		result_without_stemming = self.separator.join(values_without_stemming)

		return result_with_stemming, result_without_stemming


