import os
import csv
import sys
from utils.statement_utilities import StatementUtilities
import re
import string

from gensim import  models

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from numpy.linalg import norm
from numpy import dot

import pandas as pd

import numpy as np

ps = PorterStemmer()


nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

class CodeCommentFeatures():
    def __init__(self, separator, set, input_folder, xml_folder):
        self.separator=separator
        self.set=set
        self.header = ["id_instance", "common keywords", "cosine distance", "line distance", "statement distance"]

        self.input_folder = input_folder

        self.input_files = [f for f in os.listdir(input_folder) if ".csv" in f]

        self.xml_folder = xml_folder

        self.xml_files = [f for f in os.listdir(xml_folder) if ".csv" in f]


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

    def cosine_similarity(self, model, w1, w2):
        numerator = dot(model[w1], model[w2])
        denominator = norm(model[w1]) * norm(model[w2])
        return numerator / denominator

    def load_model(self):
        model = models.Word2Vec.load("skipgram_model.model")
        return model.wv


    def compute_similarity_comment_statement(self, model, tokens, comment_tokens, statement_tokens):

        similarity_ct=list()
        similarity_st=list()

        similarity_C_S=0
        similarity_S_C=0

        for ct in comment_tokens:
            if ct not in tokens: # that token is not present in the training tokens, we ignore it
                # print("{} not present".format(ct))
                continue

            all_sim=list()

            for st in statement_tokens:
                if st not in tokens:
                    # print("{} not present".format(st))
                    continue
                simil=self.cosine_similarity(model, ct, st)
                # print("{} {} {}".format(ct, st, simil))
                all_sim.append(simil)

            # print(all_sim)
            max_sim=0
            if len(all_sim)>0:
                max_sim=max(all_sim)
            similarity_ct.append(max_sim)


        if len(similarity_ct)>0:
            similarity_C_S=sum(similarity_ct)/len(similarity_ct)

        # print(similarity_ct)
        # print(similarity_C_S)

        for st in statement_tokens:
            if st not in tokens:  # that token is not present in the training tokens, we ignore it
                # print("{} not present".format(st))
                continue

            all_sim = list()

            for ct in comment_tokens:
                if ct not in tokens:
                    # print("{} not present".format(ct))
                    continue
                simil = self.cosine_similarity(model, st, ct)
                # print("{} {} {}".format(st, ct, simil))
                all_sim.append(simil)

            # print(all_sim)

            max_sim = 0
            if len(all_sim) > 0:
                max_sim = max(all_sim)

            similarity_st.append(max_sim)

        if len(similarity_st) > 0:
            similarity_S_C = sum(similarity_st) / len(similarity_st)

        # print(similarity_st)
        # print(similarity_S_C)

        return round((similarity_C_S+similarity_S_C)/2,2)


    def test_compute_cosine_similarity(self):
        model=self.load_model()
        # from gensim.models import KeyedVectors as word2vec
        #
        # model = word2vec.load_word2vec_format("skipgram_model.model", binary=False)

        print(type(model))
        print(dir(model))

        # print(model.wv.most_similar(positive=['equals', 'tolower']))

        tokens = (list(model.key_to_index.keys()))  # list of all tokens

        first_token = list(model.key_to_index.keys())[0]
        second_token = list(model.key_to_index.keys())[1]
        print(first_token)
        print(second_token)
        print(model.key_to_index[first_token])  # 0

        r=self.cosine_similarity(model=model, w1="get", w2="set")
        print(r)
        sys.exit(0)


    def get_distance(self, comment_statement, statements, statement):
        '''
        return the distance in lines and statements between the comment and the current statement
        if the statement spans more than one line, I will consider the minimum distance
        '''

        comment_line=comment_statement.start_line

        statement_start=statement.start_line
        statement_end=statement.end_line

        # the comment is inside the statement
        if comment_line>=statement_start and comment_line<=statement_end:
            return 0,0


        # difference in number of lines
        delta_start=abs(statement_start-comment_line)-1
        delta_end=abs(statement_end-comment_line)-1

        delta_lines=delta_start
        if delta_end<delta_start:
            delta_lines=delta_end

        if delta_lines<0:
            delta_lines=0

        keys=list(statements.keys())

        # difference in number of statements
        delta_statement=0

        if comment_line < statement_start:
            for k in range(comment_line+1, statement_start-1):
                if k in keys:
                    if statements[k].type !="EMPTY":
                        delta_statement+=1

        else:
            for k in range(statement_end+1, comment_line):
                if k in keys:
                    if statements[k].type !="EMPTY":
                        delta_statement+=1

        return delta_lines, delta_statement

    def compute_features(self, feature_dist, feature_cosine, feature_keyword):
        result=dict()

        for k in feature_dist.keys():
            if k not in feature_cosine or k not in feature_keyword:
                continue

            values=list()
            values.append(str(k))
            values.append(str(feature_keyword[k]))
            values.append(str(feature_cosine[k]))
            values.append(str(feature_dist[k].split("|")[0]))
            values.append(str(feature_dist[k].split("|")[1]))

            result[k]=self.separator.join(values)

        return result

    def clean_comment(self, comment):
        comment_list = eval(comment)
        comment_list = [c.replace("/*", "").replace("*/", "").replace("//", "").strip() for c in comment_list]
        return " ".join(comment_list)


    def process_files(self):

        for file in self.input_files:

            if self.set not in file:
                continue

            features_distance=dict()
            features_cosine=dict()
            features_common_keyword=dict()

            dataset=file.split(".")[0]

            lines=pd.read_csv(os.path.join(self.input_folder, file))

            lines_xml=pd.read_csv(os.path.join(self.xml_folder, "{}.csv".format(dataset)))

            xml_dict=dict() # xml with the single comment in the code (we kept only the comment we associated to the code)
            for index, line in lines_xml.iterrows():
                id_instance = int(line["id"])
                print(file, id_instance)

                xml_code = line["xml_single_comment"]
                print(xml_code)
                xml_code = "\n".join(eval(xml_code))
                xml_dict[id_instance]=xml_code

            xml_dict_no_comments=dict() # xml without the comments
            for index, line in lines_xml.iterrows():
                id_instance = int(line["id"])

                xml_code = line["xml"]
                xml_code = "\n".join(eval(xml_code))
                xml_dict_no_comments[id_instance]=xml_code


            errors = 0

            # word2vec model and list of tokens in the word2vec model
            model = self.load_model()
            all_tokens_word2vec = (list(model.key_to_index.keys()))  # list of all tokens

            print("CODE COMMENT FEATURES {} LINES".format(len(lines)))

            for index, line in lines.iterrows():
                try:

                    id_instance = int(line["index"])

                    print(id_instance)

                    if id_instance not in xml_dict.keys():
                        print("{} SKIPPED".format(id_instance))
                        continue

                    # R2: semantic similarity

                    xml_code = xml_dict_no_comments[id_instance]
                    comment=self.clean_comment(line["comment"])
                    tokens_comment=self.process_text(comment)
                    # print(tokens_comment)

                    # print(self.cosine_similarity(model, all_tokens_word2vec, tokens_comment[0], tokens_comment[1]))

                    statements = dict()
                    su = StatementUtilities()

                    statements = su.return_statements(statements, xml_code)

                    dict_statements=dict()

                    for k in statements.keys():

                        if statements[k].type=="EMPTY":
                            continue

                        # statements[k].print_statement()

                        key_feature="{}_{}".format(id_instance, statements[k].start_line)


                        # statements[k].print_statement()
                        code=statements[k].code
                        tokens_statement=self.process_text(code)

                        # we use the code without spaces and new lines as a key for mapping the statements without comments
                        # and the statements with the comment we're looking for
                        # it works almost everytime
                        key_dict=code.replace(" ","").replace("\n","")
                        if key_dict not in dict_statements:
                            dict_statements[key_dict]=list()
                        dict_statements[key_dict].append(key_feature)

                        similarity=self.compute_similarity_comment_statement(model, all_tokens_word2vec, tokens_comment, tokens_statement)
                        # print(similarity)
                        features_cosine[key_feature]=similarity
                        # we initialize the other dicts to 0, we can replace them later on with the right value
                        features_common_keyword[key_feature]=-1
                        features_distance[key_feature]="0|0"

                    # print(features_cosine)
                    # print(dict_statements)
                    # R1, R3 and R4 features: distance in line and statement between the comment and each statement
                    # and common key word

                    xml_code = xml_dict[id_instance]

                    comment=self.clean_comment(line["comment"])
                    tokens_comment=self.process_text(comment)

                    statements = dict()

                    su = StatementUtilities()

                    statements = su.return_statements(statements, xml_code)

                    # for k in statements.keys():
                    #     statements[k].print_statement()

                    comment_statement=None
                    for k in statements.keys():
                        if statements[k].type=="comment":
                            comment_statement=statements[k]

                    for k in statements.keys():
                        # statements[k].print_statement()

                        if statements[k].type=="EMPTY":
                            continue

                        if statements[k].type=="comment":
                            continue

                        code=statements[k].code

                        code_no_space=code.replace(" ","").replace("//test","").replace("\n","")
                        # print(code_no_space)
                        if code_no_space not in dict_statements.keys():
                            print("KEY NOT FOUND")
                            continue
                        if len(dict_statements[code_no_space])==0:
                            continue
                        key_feature=dict_statements[code_no_space][0]
                        dict_statements[code_no_space].pop(0)

                        tokens_statement=self.process_text(code)
                        # print(tokens_statement)
                        # print(tokens_comment)
                        rr=list(set(tokens_comment) & set(tokens_statement))

                        num_keywords=len(rr)

                        dist1, dist2=self.get_distance(comment_statement, statements, statements[k])
                        features_common_keyword[key_feature]=num_keywords
                        features_distance[key_feature]="{}|{}".format(dist1, dist2)

                    # sys.exit(0)

                    # print(features_distance)
                    # print(features_common_keyword)

                except Exception as e:
                    errors += 1
                    print("ERROR__ERROR")
                    print(e)
                    sys.exit(0)

            print("{} ERRORS".format(errors))


            tot=0
            for k in features_common_keyword.keys():
                if features_common_keyword[k]==-1:
                    print(k)
                    tot+=1

            print("{} anomalies".format(tot))

            features=self.compute_features(features_distance, features_cosine, features_common_keyword)
            print(len(features_distance.keys()))
            return self.header, features