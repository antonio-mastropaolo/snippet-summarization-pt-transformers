
import pandas as pd
import string
import re
import sys

import nltk
from nltk.corpus import stopwords

import pandas as pd

from nltk.stem import PorterStemmer

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

def tokenize_line(line):

    # comment="the programmer programmed a new file"

    text_new = line

    for character in string.punctuation:
        text_new = text_new.replace(character, ' ')

    tokens=text_new.split(" ")

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

    res=list()
    for w in filtered_tokens:
        res.append(ps.stem(w))

    filtered_tokens=res

    # print(res)
    return filtered_tokens


def find_comment(lines):
    start_comment=list()
    end_comment=list()

    for i, l in enumerate(lines):
        if "<comment>" in l:
            start_comment.append(i)
        if "</comment>" in l:
            end_comment.append(i)

    commented_lines=list()
    for x,y in zip(start_comment, end_comment):
        for i in range(x,y+1):
            commented_lines.append(i)
    # print(start_comment, end_comment)
    # print(commented_lines)
    # print("_____")
    return commented_lines

def compute_similarity(s1, s2):
    len_s1=len(s1)
    len_s2=len(s2)

    s2_cp=s2.copy()

    numerator=0

    for e in s1:
        if e in s2_cp:
            s2_cp.remove(e)
            numerator+=1

    denominator=max(len_s1, len_s2)

    return numerator/denominator



def add_start_end(code_lines, commented_lines):

    start_lines=list()
    end_lines=list()

    # if the model did not predict any start end we return the original method
    if len(commented_lines)==0:
        return "\n".join(code_lines)

    max_lines=max(commented_lines)+2

    added_lines=dict()

    for c in commented_lines:
        # print("PROCESSNG {}".format(c))
        if c not in added_lines.keys():
            # print("ADD START")
            start_lines.append(c)
            added_lines[c]=1
            for i in range(c+1,max_lines+2):
                # print("ANALYZING {}".format(i))
                if i in commented_lines:
                    added_lines[i]=1
                else:
                    end_lines.append(i-1)
                    # print("ADD END")
                    break


    lines=code_lines


    for x,y in zip(start_lines, end_lines):
        line_s=lines[x]
        line_s="<start>"+line_s
        lines[x]=line_s

        line_e=lines[y]
        line_e=line_e+"<end>"
        lines[y]=line_e

    return "\n".join(lines)

def Average(lst):
    return sum(lst) / len(lst)


def main():
    file="test.csv"
    file_no_comments="result_authors.csv"


    df=pd.read_csv(file)
    df2=pd.read_csv(file_no_comments)

    dict_commented_lines=dict()
    dict_comment_tokens=dict()
    dict_num_lines=dict()
    dict_correct_linking=dict()

    dict_index_instance=dict()

    for index, row in df.iterrows():

        method=row["linkingInstance"]
        lines=method.split("\n")

        lines=[l.replace("<start>","").replace("<end>","") for l in lines]

        # for l in lines:
        #     print(l)

        commented_lines=find_comment(lines)
        # print(len(commented_lines))
        # print(commented_lines)

        # commented text

        comment_text=""
        for c in commented_lines:
            comment_text+="{} ".format(lines[c])

        comment_text=comment_text.replace("<comment>","").replace("</comment>","").strip()
        # print(comment_text)

        tokens_comment=tokenize_line(comment_text)

        dict_comment_tokens[index]=tokens_comment
        dict_commented_lines[index]=commented_lines
        dict_num_lines[index]=len(lines)
        dict_index_instance[index]=row["index"]

    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    dict_threshold=dict()
    dict_referred_lines=dict()

    for t in thresholds:

        print("Threshold {}".format(t))

        dict_result=dict()
        dict_ref_curr=dict()

        for index, row in df2.iterrows():

            if index%100==0:
                print("{} OUT OF {}".format(index, len(df2.index)))

            referred_lines=list()

            method=row["predicted linking"]
            correct=row["correct linking"]
            dict_correct_linking[index]=correct

            method=method.replace("<start>","").replace("<end>","").replace("//comment","")

            lines=method.split("\n")

            if len(lines) != dict_num_lines[index]:
                print("ERROR NUM LINES")
                method=row["predicted linking"]

                method=method.replace("<start>","").replace("<end>","")

                dict_result[index]=method
                dict_ref_curr[index]=list()

                continue

            commented_lines=dict_commented_lines[index]
            tokens_comment=dict_comment_tokens[index]

            # for l in lines:
            #     print(l)

            for i, l in enumerate(lines):
                if i in commented_lines:
                    continue

                tokens_line=tokenize_line(l.strip())
                # print(l)
                # print(tokens_line)
                # print(tokens_comment)
                sim=compute_similarity(tokens_line, tokens_comment)
                # print(sim)
                if sim>t:
                    referred_lines.append(i)


            method=row["predicted linking"]

            method=method.replace("<start>","").replace("<end>","")

            lines=method.split("\n")

            res=add_start_end(lines, referred_lines)

            dict_result[index]=res
            dict_ref_curr[index]=referred_lines


        dict_threshold[t]=dict_result
        dict_referred_lines[t]=dict_ref_curr


    for t in thresholds:
        res_curr=dict_threshold[t]

        col1=list()
        col2=list()
        col3=list()
        for k in res_curr.keys():
            col1.append(dict_index_instance[k])
            col2.append(res_curr[k])
            col3.append(dict_correct_linking[k])
        #
        df = pd.DataFrame({
            'id_istance': col1,
            'predicted linking': col2,
            'correct linking': col3
        })
        df.to_csv('result_textual_similarity_{}.csv'.format(t), index=False, header=True)


    for t in thresholds:
        res_curr=dict_referred_lines[t]

        ref_lines_len=list()


        curr=0
        tot=0

        for k in res_curr.keys():
            if len(res_curr[k])>0:
                curr+=1
                ref_lines_len.append(len(res_curr[k]))
                # print(res_curr[k])
                # print(ref_lines_len)
            tot+=1

        print("Threshold {}: {} OUT OF {}".format(t, curr, tot))
        print("average num lines: {}".format(Average(ref_lines_len)))




if __name__=="__main__":
    main()