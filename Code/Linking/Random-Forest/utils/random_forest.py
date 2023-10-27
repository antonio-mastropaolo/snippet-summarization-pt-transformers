from sklearn.ensemble import RandomForestClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
import csv
import numpy as np
import math
import json
from datetime import datetime
import os
from sklearn.metrics import f1_score

import joblib

import sys


n_estimators=[100,200,400,600,800,1000]
criterions=["gini", "entropy"]
max_depths=[10,15,20,40,60]
max_features=[5,10,15]
class_1_weights=[1,2,3,4,5,6,7]

n_estimators=[100,200,300]
criterions=["gini"]
max_depths=[15,25]
max_features=[5,10,15]
class_1_weights=[1,3,5]


class RandomForest():
    def __init__(self, output_folder):
        self.output_folder=output_folder
        self.train_file=os.path.join(output_folder, "train_features.csv")
        self.train_result=os.path.join(output_folder, "train_results.csv")

        self.test_file=os.path.join(output_folder, "test_features.csv")
        self.test_result=os.path.join(output_folder, "test_results.csv")

        self.eval_file=os.path.join(output_folder, "eval_features.csv")
        self.eval_result=os.path.join(output_folder, "eval_results.csv")

    def read_data(self, data_path):
        file = open(data_path, 'r')
        csvreader = csv.reader(file)

        rows = []
        for row in csvreader:
            rows.append(row)

        file.close()

        print("Read {} lines".format(len(rows)))
        return rows

    def create_X_y(self, X_full, y_full):
        '''
        create the X and the y for trainig the model
        We have to ignore the first 4 columns of the X and the first column of y
        '''
        res_X=list()
        for i,x in enumerate(X_full[1:]):
            res_X.append(x[4:])

        X_np = np.asarray(res_X)
        print(X_np.shape)

        res_y=list()
        for i,y in enumerate(y_full[1:]):
            res_y.append(y[1:])

        y_np = np.asarray(res_y)

        y_np=np.ravel(y_np) # we need a np array of (xxx,) and not (xxx,1)
        return X_np, y_np

    def train_random_forest_authors(self):
        '''
        In the paper the authors say:
        we set the number of decision trees in the random forests algorithm as 150 and the number of features to use in
        random se- lection as 13 to obtain the best model
        Hence we kept the default values for all the other parameters and we set the parameters suggested by the authors
        '''
        X_full=self.read_data(self.train_file)
        y_full=self.read_data(self.train_result)

        X,y=self.create_X_y(X_full, y_full)

        clf = RandomForestClassifier(n_estimators=150, max_features=13, random_state=0)

        clf.fit(X, y)

        num_0=0
        num_1=1

        correct=0
        correct_1=0
        total_1=0

        X_test_full=self.read_data(self.test_file)
        y_test_full=self.read_data(self.test_result)

        print(len(X_test_full))

        X_test,y_test=self.create_X_y(X_test_full, y_test_full)

        chunk_size=100
        num_chunks=math.ceil(len(y_test)/100)

        y_true=list()
        y_pred=list()

        for i in range(num_chunks):
            print(i)
            res = clf.predict(X_test[i*100:(i+1)*100, :])
            correct_value = y_test[i*100:(i+1)*100]

            for xx, yy in zip(res, correct_value):

                y_true.append(int(yy))
                y_pred.append(int(xx))

                is_correct = False

                if xx==str(yy):
                    is_correct=True

                if yy=="1":
                    total_1+=1

                if is_correct:
                    correct+=1

                if is_correct and yy=="1":
                    correct_1+=1


                if xx!="0":
                    num_1+=1
                else:
                    num_0+=1


        print("{} OUT OF {}".format(correct, len(y_test)))
        print("1: {} OUT OF {}".format(correct_1, total_1))

        score=((correct/len(y_test))+(correct_1/total_1))/2

        print(f1_score(y_true, y_pred, average='binary'))

        file=open("{}/prediction_model_authors.txt".format(self.output_folder), "w+")

        print(len(y_test_full[1:]))
        print(len(y_pred))

        file.write('id_istance|||Prediction\n')
        for x, y in zip(y_test_full[1:], y_pred):
            file.write('{}|||{}\n'.format(x[0],y))

        file.close()
        print(score)

    def train_random_forest(self, best_config):
        X_full=self.read_data(self.train_file)
        y_full=self.read_data(self.train_result)

        X,y=self.create_X_y(X_full, y_full)

        parts=best_config.split("|")
        n_estimator=int(parts[0])
        criterion=parts[1]
        max_depth=int(parts[2])
        max_feature=int(parts[3])
        class_1_weight=int(parts[4])

        clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion,
                                     max_depth=max_depth, max_features=max_feature,
                                     class_weight={"0": 1, "1": class_1_weight}, random_state=0)



        clf.fit(X, y)

        num_0=0
        num_1=1

        correct=0
        correct_1=0
        total_1=0

        X_test_full=self.read_data(self.test_file)
        y_test_full=self.read_data(self.test_result)

        print(len(X_test_full))

        X_test,y_test=self.create_X_y(X_test_full, y_test_full)

        chunk_size=100
        num_chunks=math.ceil(len(y_test)/100)

        y_true=list()
        y_pred=list()

        for i in range(num_chunks):
            print(i)
            res = clf.predict(X_test[i*100:(i+1)*100, :])
            correct_value = y_test[i*100:(i+1)*100]

            for xx, yy in zip(res, correct_value):

                y_true.append(int(yy))
                y_pred.append(int(xx))

                is_correct = False

                if xx==str(yy):
                    is_correct=True

                if yy=="1":
                    total_1+=1

                if is_correct:
                    correct+=1

                if is_correct and yy=="1":
                    correct_1+=1


                if xx!="0":
                    num_1+=1
                else:
                    num_0+=1


        print("{} OUT OF {}".format(correct, len(y_test)))
        print("1: {} OUT OF {}".format(correct_1, total_1))

        score=((correct/len(y_test))+(correct_1/total_1))/2

        print(f1_score(y_true, y_pred, average='binary'))


        file=open("prediction_model.txt", "w+")

        print(len(y_test_full[1:]))
        print(len(y_pred))

        file.write('id_istance|||Prediction\n')
        for x, y in zip(y_test_full[1:], y_pred):
            file.write('{}|||{}\n'.format(x[0],y))

        file.close()

        print(score)


    def hyper_parameter_tuning(self):
        '''
        test different configurations to find the best model on eval set
        '''

        X_full=self.read_data(self.train_file)
        y_full=self.read_data(self.train_result)

        X,y=self.create_X_y(X_full, y_full)

        print(datetime.now())

        dict_result_0=dict() # accuracy on 0
        dict_result_1=dict() # accuracy on 1
        dict_result_both=dict() # average of the accuracy

        dict_f1=dict()

        total=len(n_estimators)*len(criterions)*len(max_depths)*len(max_features)*len(class_1_weights)

        curr=0

        for n_estimator in n_estimators:
            for criterion in criterions:
                for max_depth in max_depths:
                    for max_feature in max_features:
                        for class_1_weight in class_1_weights:

                            print("PROCESSNG {} OUT OF {}".format(curr+1, total))

                            key="{}|{}|{}|{}|{}".format(n_estimator, criterion, max_depth, max_feature, class_1_weight)

                            clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion,
                            max_depth=max_depth, max_features=max_feature, class_weight={"0": 1, "1": class_1_weight},random_state=0)

                            clf.fit(X, y)

                            num_0=0
                            num_1=1

                            correct=0
                            correct_1=0
                            total_1=0

                            y_true=list()
                            y_pred=list()

                            X_test_full=self.read_data(self.eval_file)
                            y_test_full=self.read_data(self.eval_result)


                            X_test,y_test=self.create_X_y(X_test_full, y_test_full)

                            chunk_size=100
                            num_chunks=math.ceil(len(y_test)/100)

                            for i in range(num_chunks):
                                print(i)
                                res = clf.predict(X_test[i*100:(i+1)*100, :])
                                correct_value = y_test[i*100:(i+1)*100]

                                for xx, yy in zip(res, correct_value):

                                    y_true.append(int(yy))
                                    y_pred.append(int(xx))

                                    is_correct = False

                                    if xx==str(yy):
                                        is_correct=True

                                    if yy=="1":
                                        total_1+=1

                                    if is_correct:
                                        correct+=1

                                    if is_correct and yy=="1":
                                        correct_1+=1


                                    if xx!="0":
                                        num_1+=1
                                    else:
                                        num_0+=1




                            print("{} OUT OF {}".format(correct, len(y_test)))
                            print("1: {} OUT OF {}".format(correct_1, total_1))

                            score=((correct/len(y_test))+(correct_1/total_1))/2

                            print(score)

                            dict_f1[key]=f1_score(y_true, y_pred, average='binary')
                            # dict_f1[key]=score

                            print(datetime.now())

                            dict_result_0[key]=(correct/len(y_test))
                            dict_result_1[key]=(correct_1/total_1)
                            dict_result_both[key]=score

                            curr+=1


        a_file = open("../data_both.json", "w")
        json.dump(dict_result_both, a_file)
        a_file.close()

        a_file = open("../data_0.json", "w")
        json.dump(dict_result_0, a_file)
        a_file.close()

        a_file = open("../data_1.json", "w")
        json.dump(dict_result_1, a_file)
        a_file.close()

        a_file = open("../data_f1score.json", "w")
        json.dump(dict_f1, a_file)
        a_file.close()

        return dict_f1

    def test(self):

        X, y = make_classification(n_samples=1000, n_features=4,
                                    n_informative=2, n_redundant=0,
                                    random_state=0, shuffle=False)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        print(type(X))
        print(type(y))
        print(X.shape)
        print(clf.predict([[0, 0, 0, 0]]))

    def test2(self):
        from numpy import mean
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedStratifiedKFold
        from imblearn.ensemble import BalancedRandomForestClassifier
        # generate dataset
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
        # define model
        model = BalancedRandomForestClassifier(n_estimators=10)
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        # summarize performance
        print('Mean ROC AUC: %.3f' % mean(scores))

    def test3(self):
        from numpy import mean
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedStratifiedKFold
        from imblearn.ensemble import EasyEnsembleClassifier
        # generate dataset
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
        # define model
        model = EasyEnsembleClassifier(n_estimators=10)
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        # summarize performance
        print('Mean ROC AUC: %.3f' % mean(scores))
