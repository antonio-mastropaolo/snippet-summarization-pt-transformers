import pandas as pd
import os

class ExportDataset():
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def export_dataset(self):
        sets=["train", "eval", "test"]

        for s in sets:

            data=pd.read_csv("{}_results.csv".format(s), index_col=0)
            print(len(data.index))

            print(list(data.index)[:100])

            data=pd.read_csv("{}_features.csv".format(s), index_col=0)
            print(len(data.index))

            print(list(data.index)[:100])

            dataset_path=os.path.join(self.input_folder, "{}-linking.csv")

            dataset=pd.read_csv(dataset_path.format(s), index_col=1)
            print(len(dataset.index))

            indexes=list(dataset.index)

            min_=min(indexes)
            max_=max(indexes)

            print(min_, max_)

            dict_index=dict()
            for i in indexes:
                dict_index[i]=1

            # for i in range(min_, max_):
            #     if i not in dict_index.keys():
            #         print("NOT FOUND {}".format(i))

            indexes_features=list(data.index)

            for i in indexes_features:
                instance=int(i.split("_")[0])
                dict_index[instance]=0

            to_remove=list()
            for k in dict_index.keys():
                if dict_index[k]==1:
                    to_remove.append(k)

            print(to_remove)

            print("BEFORE {} INDEX".format(len(dataset.index)))
            print("REMOVED {} ROWS".format(len(to_remove)))
            dataset_filtered=dataset.drop(to_remove)
            print("AFTER {} INDEX".format(len(dataset_filtered.index)))

            dataset_filtered.to_csv("{}_filtered.csv".format(s))
