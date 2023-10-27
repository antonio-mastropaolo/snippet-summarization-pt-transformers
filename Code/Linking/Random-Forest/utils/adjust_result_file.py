import os
import re
import sys
import pandas as pd

class AdjustResultFile():
    def __init__(self):
        pass


    def adjust(self):
        df=pd.read_csv("result.csv")

        col1=list()
        col2=list()
        col3=list()

        for index, row in df.iterrows():
            pl=row["predicted linking"]
            print(pl)
            lines=pl.split("\n")
            start_lines=list()
            end_lines=list()
            for i, l in enumerate(lines):
                if "<start>" in l:
                    start_lines.append(i)
                if "<end>" in l:
                    end_lines.append(i)

            print(start_lines)
            print(end_lines)

            # check if we can have two statements referred by a comment that are consecutive
            # e.g. from 3 to 5 and from 6 to 7.
            # if this is not happening I'm FINE and I can simply keep the first occurrence of start and end
            for end in end_lines:
                if end+1 in start_lines:
                    print("ERROR")
                    sys.exit(0)

            pl_new=pl.replace("<start>", "<START>",1).replace("<end>", "<END>",1)
            pl_new=pl_new.replace("<start>", "").replace("<end>", "")
            pl_new=pl_new.replace("<START>", "<start>").replace("<END>", "<end>")


            col1.append(row["id_istance"])
            col2.append(pl_new)
            col3.append(row["correct linking"])
        #
        df = pd.DataFrame({
            'id_istance': col1,
            'predicted linking': col2,
            'correct linking': col3
        })
        df.to_csv('result_adjusted.csv', index=False, header=True)



