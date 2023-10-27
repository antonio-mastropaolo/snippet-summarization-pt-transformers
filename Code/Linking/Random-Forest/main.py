
from features.code_features import CodeFeatures
from features.comment_features import CommentFeatures
from features.code_comment_features import CodeCommentFeatures
import csv
from utils.skipgram_train import SkipGram
from utils.random_forest import RandomForest
from utils.create_predicted_methods import CreatePredictedMethod
from utils.export_dataset import ExportDataset
from utils.adjust_result_file import AdjustResultFile

import sys

import argparse

# LOCAL
# original dataset
input_folder="dataset"
# xml dataset processed in step 1
xml_folder="xml"
# output folder where we save the skipgram model and the feature vectors
output_folder="output"

class Wrapper():
    def __init__(self):
        self.separator = "(*&^*("


    def write_csv(self, header, features, name_file):

        with open(name_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            for key in features:
                feature=features[key]
                parts = feature.split(self.separator)

                # xml = dict_data[k]
                # data = list()
                # data.append(str(k))
                # data.append(xml)

                # write the data
                writer.writerow(parts)

    def write_is_commented(self, features_full, is_commented, name_file):

        header=["id_instance", "is_commented"]

        with open(name_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            for key in is_commented.keys():
                if key not in features_full.keys():
                    continue
                values=list()
                values.append(str(key))
                values.append(str(is_commented[key]))

                writer.writerow(values)


    def merge_features(self, code_feature, comment_feature, code_comment_feature,
                       header_code, header_comment, header_code_comment):
        result_features=dict()

        header=header_code
        header.extend(header_comment[1:])
        header.extend(header_code_comment[1:])

        features=dict()

        for key in code_feature.keys():

            id_instance=int(key.split("_")[0])

            if id_instance not in comment_feature.keys():
                continue

            if key not in code_comment_feature.keys():
                continue

            curr_value_code=code_feature[key].split(self.separator)
            curr_value_comment=comment_feature[id_instance].split(self.separator)
            curr_value_code_comment=code_comment_feature[key].split(self.separator)

            res=curr_value_code
            res.extend(curr_value_comment[1:])
            res.extend(curr_value_code_comment[1:])

            # print(res)

            features[key]=self.separator.join(res)

        return header, features


    def write_all_features(self):

        sets=["train", "eval", "test"]

        for s in sets:

            code_feature_class=CodeFeatures(self.separator, s, xml_folder)
            header_code, feature_code, is_commented=code_feature_class.process_files()

            comment_feature_class=CommentFeatures(self.separator, s, input_folder)

            header_comment, feature_comment_with_stemming, feature_comment_without_stemming =comment_feature_class.process_files()

            code_comment_feature_class=CodeCommentFeatures(self.separator, s, input_folder, xml_folder)

            header_code_comment, feature_code_comment =code_comment_feature_class.process_files()

            # print(len(feature_code_comment.keys()))

            header_full, feature_full=self.merge_features(feature_code, feature_comment_with_stemming, feature_code_comment,
                                                          header_code, header_comment, header_code_comment)

            self.write_csv(header_full, feature_full, "{}/{}_features.csv".format(output_folder, s))

            self.write_is_commented(feature_full, is_commented, "{}/{}_results.csv".format(output_folder, s))

    def train_skipgram_model(self):
        s = SkipGram(input_folder)
        s.train_skipgram_model()

    def train_random_forest(self):
        rf = RandomForest(output_folder)

        # train the random forest with the parameters defined by the authors
        rf.train_random_forest_authors()

    def create_result_files(self):
        cpm = CreatePredictedMethod(xml_folder, output_folder, input_folder)
        cpm.create_output_file()
        # cpm.find_all_commments()


    def adjust_result_files(self):
        arf=AdjustResultFile()
        arf.adjust()

    def export_dataset(self):
        e=ExportDataset(input_folder)
        e.export_dataset()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-train_skipgram", "--train_skipgram", action="store_true",
                        help="train the skipgram model to create code comment features")

    parser.add_argument("-write_all_features", "--write_all_features", action="store_true",
                        help="create the features for creating the features")

    parser.add_argument("-train_random_forest", "--train_random_forest", action="store_true",
                        help="hyper parameter tuning for finding the best random forest config on the eval set; then using the config from the authors")

    parser.add_argument("-save_results", "--save_results", action="store_true",
                        help="save the final files")

    parser.add_argument("-adjust_results", "--adjust_results", action="store_true",
                        help="adjust the result files")

    parser.add_argument("-export_dataset", "--export_dataset", action="store_true",
                        help="save the final files")

    args = parser.parse_args()

    w=Wrapper()

    if args.train_skipgram:
        # step1: train the skipgram model to create code comment features
        w.train_skipgram_model()
    if args.write_all_features:
        # step2: create the features for creating the features
        w.write_all_features()
    if args.train_random_forest:
        # step3: hyper parameter tuning for finding the best random forest config on the eval set
        # then using the config from the authors
        # saving the results
        w.train_random_forest()
    if args.save_results:
        # step4: save the final files
        w.create_result_files()

    if args.adjust_results:
        # step4bis: adjust the result files
        # since the model was used to comment a block of continuous lines, we only consider the first group of consecutive lines
        # to be commented, ignoring all the other lines that the model was recommending to comment
        w.adjust_result_files()

    if args.export_dataset:
        # step5: export the dataset with the instances used for training/evaluating the random forest
        w.export_dataset()

if __name__=="__main__":
    main()


