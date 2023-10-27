import csv
from utils.statement_utilities import StatementUtilities
import xml.etree.ElementTree as ET
import os
import re
import sys
import pandas as pd

# xml_folder = "/Users/matteo.ciniselli/OneDrive - USI/code/007_Comment_Labeling/out/1_baseline/test.csv"


class CreatePredictedMethod():
    def __init__(self, xml_folder, output_folder, input_folder):
        self.xml_folder=xml_folder
        self.xml_file=os.path.join(xml_folder, "test.csv")
        self.output_folder=output_folder
        self.input_folder=input_folder

        self.commented_lines=os.path.join(input_folder, "commented_lines.txt")


    def read_data(self, data_path):
        file = open(data_path, 'r')
        csvreader = csv.reader(file)

        rows = []
        for row in csvreader:
            rows.append(row)

        file.close()

        print("Read {} lines".format(len(rows)))
        return rows

    def comment_lines(self, code_lines, commented_lines):

        start_lines=list()
        end_lines=list()

        # if the model did not predict any start end we return the original method
        if len(commented_lines)==0:
            return code_lines

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


        lines=code_lines.split("\n")

        for x,y in zip(start_lines, end_lines):
            line_s=lines[x-1]
            line_s="<start>"+line_s
            lines[x-1]=line_s

            line_e=lines[y-1]
            line_e=line_e+"<end>"
            lines[y-1]=line_e

        return "\n".join(lines)

    def generate_result_random_forest(self, random_forest_output, dict_code, authors_start_end, correct_start_end):
        path_test_file = os.path.join(self.output_folder, "test_features.csv")

        features = self.read_data(path_test_file)
        result_random_forest = dict()

        file = open(random_forest_output, "r")

        lines = file.readlines()
        lines = [l.strip() for l in lines]

        file.close()

        for l in lines[1:]:
            parts = l.split("|||")
            result_random_forest[parts[0]] = int(parts[1])

        referred_lines = dict()
        for feature in features[1:]:
            id_statement = feature[0]

            id_instance = int(id_statement.split("_")[0])

            start_line = int(feature[1])
            end_line = int(feature[2])

            if id_instance not in referred_lines.keys():
                referred_lines[id_instance] = list()

            if result_random_forest[id_statement] == 1:
                for i in range(start_line, end_line + 1):
                    referred_lines[id_instance].append(i)

        # remove duplicates and sort values
        for k in referred_lines.keys():
            curr = referred_lines[k]
            curr = list(set(curr))

            curr.sort()
            referred_lines[k] = curr

        for k in dict_code.keys():
            code = dict_code[k]

            # if some error occured during the creation features/train random forest we ignore it
            if k not in referred_lines.keys():
                continue
            commented_lines = referred_lines[k]

            lines = self.comment_lines(code, commented_lines)
            authors_start_end[k] = lines

        tot_correct = 0
        tot = 0

        for k in correct_start_end.keys():
            # print(k)
            # print(correct_start_end[k])
            # print(authors_start_end[k])

            if k not in authors_start_end.keys():
                print("NOT FOUND")
                continue

            if correct_start_end[k].replace(" ", "") == authors_start_end[k].replace(" ", ""):
                tot_correct += 1
            tot += 1

        print("{} OUT OF {}".format(tot_correct, tot))

        return authors_start_end

    def find_all_commments(self):

        result_name = os.path.join(self.input_folder, "test.csv")

        rows=pd.read_csv(result_name)

        f=open(self.commented_lines, "r")

        lines=f.readlines()
        f.close()

        comm=eval(lines[0].strip())

        dict_comments=dict()
        for k in comm.keys():
            dict_comments[int(k)]=eval(comm[k])

        # print(dict_comments[1808])
        # sys.exit(0)

        # print(dict_comments)


        dict_len_methods=dict()

        for i, l in rows.iterrows():
            if i % 1 == 0:
                print("PROCESSED {} OUT OF {}".format(i, len(rows.index)))

            name = int(l["index"])

            if name not in dict_comments:
                dict_comments[name]=list()

            code = l["originalMethod"]
            print(code)
            num_lines_method=len(code.split("\n"))
            # print(num_lines_method)
            dict_len_methods[name]=num_lines_method

            continue

            # print(code)
            code_no_space=code.replace(" ","")
            print(code_no_space)

            # print(code)

            comments=list()

            result = re.finditer(comment_regex, code_no_space)

            lines=code.split("\n")

            for match_obj in result:
                # print(match_obj.lastindex)
                # print(match_obj.lastgroup)
                st=(match_obj.start())
                end=(match_obj.end())
                comment=code_no_space[st:end]
                f=False
                if comment[0]=="\n":
                    comment=comment[1:]
                    st=st+1
                if comment[-1]=="\n":
                    comment=comment[:-1]
                    end=end-1

                if code_no_space[st-1] != "\n":

                    continue
                else:
                    comments.append(match_obj.group())
                    # we can find the line by counting the number of new lines
                    line_commented=len(code_no_space[:st].split("\n"))
                    dict_comments[name].append(line_commented)
                    lines[line_commented-1]="<<AA>>"


            code="\n".join(lines)
            code_new = re.sub(comment_regex, "", code)
            # print(code_new)

            code_no_space=code_new.replace(" ","")
            print(code_no_space)

            result = re.finditer(comment_regex2, code_no_space)

            for match_obj in result:

                st = (match_obj.start())
                end = (match_obj.end())


                if comment[0]=="\n":
                    comment=comment[1:]
                    st=st+1
                if comment[-1]=="\n":
                    comment=comment[:-1]
                    end=end-1

                print(st)
                print(end)

                print(code_no_space[st:end])

                num_lines_comment=len(code_no_space[st:end].split("\n"))

                parts=code_no_space[st:end].split("\n")
                print("AAA")
                for p in parts:
                    print(p)

                print("{} LINES".format(num_lines_comment))

                # if the previous char is not a new lines we can ignore the first line
                # because we already have a non empty line
                if code_no_space[st - 1] != "\n":
                    num_lines_comment-=1
                    print("XXXXX")

                # if the next char is not a new lines we can ignore the last line
                # because we already have a non empty line
                if code_no_space[end] != "\n":
                    num_lines_comment-=1
                    print("YYYYY")

                print("{} COMMENTS".format(num_lines_comment))

                if num_lines_comment<=0:
                    continue

                comments.append(match_obj.group())

                start_line_commented = len(code_no_space[:st].split("\n"))

                for i in range(start_line_commented, start_line_commented+num_lines_comment):
                    dict_comments[name].append(i)




        return dict_comments, dict_len_methods


    def inject_comments(self, dict_comments, dict_len_methods, key, method):

        if key not in dict_comments.keys():
            print("ERROR")

        commented_lines=dict_comments[key]
        len_method=dict_len_methods[key]
        print(len_method)
        print(commented_lines)

        lines=method.split("\n")
        result_lines=list()
        for l in lines:
            result_lines.append(l)
            curr_lengh=len(result_lines)
            if curr_lengh + 1 in commented_lines:
                result_lines.append("//comment")
                curr_lengh=len(result_lines)
                print("CURR {}".format(curr_lengh))
                for i in range(curr_lengh, 2000): # 2000 is a very high number, just to add all the conseutive commented lines
                    if i+1 in commented_lines:
                        result_lines.append("//comment")
                    else:
                        break


        print("KEY: {}".format(key))
        for i, l in enumerate(result_lines):
            print(i+1,l)
        print(len(result_lines))

        error=False

        if len(result_lines) != len_method:

            error=True


        return "\n".join(result_lines), error

    def create_output_file(self):
        '''
        create a csv with each method with the correct <start><end> (based on the real commented lines)
        and the predicted <start><end> (based on the random forest)
        '''

        xml_lines=pd.read_csv(self.xml_file)

        su = StatementUtilities()

        # this dict contains the correct method with the real start end tags
        correct_start_end=dict()

        dict_comments, dict_len_methods=self.find_all_commments()

        # print(dict_comments)
        # print(dict_len_methods)

        dict_code=dict()


        for index, line in xml_lines.iterrows():
            id_instance = int(line["id"])

            xml_code = line["xml"]
            xml_code = "\n".join(eval(xml_code))

            tree = ET.fromstring(xml_code)

            code = su.get_text((tree))
            dict_code[id_instance]=code

            commented_lines=eval(line["referenced_lines"])

            lines=self.comment_lines(code, commented_lines)
            correct_start_end[id_instance]=lines

        # create the dict with the start end predicted by the random forest of the authors

        # this dict contains the method with start and end predicted by the random forest of the authors
        authors_start_end=dict()
        path_prediction_authors = os.path.join(self.output_folder, "prediction_model_authors.txt")

        authors_start_end=self.generate_result_random_forest(path_prediction_authors, dict_code, authors_start_end, correct_start_end)

        authors_start_end_injected=dict()

        errors_author=list()
        errors_correct=list()

        for k in authors_start_end.keys():
            method_curr=authors_start_end[k]
            method_final, error=self.inject_comments(dict_comments, dict_len_methods, k, method_curr)
            if len(method_final)>0:
                authors_start_end_injected[k]=method_final
            if error:
                errors_author.append(k)

        correct_start_end_injected=dict()
        for k in authors_start_end_injected.keys():
            method_curr=correct_start_end[k]
            method_final, error=self.inject_comments(dict_comments, dict_len_methods, k, method_curr)
            correct_start_end_injected[k]=method_final
            if error:
                errors_correct.append(k)

        # df = pd.DataFrame.from_dict(authors_start_end_injected)
        # df.to_csv('result_authors.csv', index=False, header=True)

        col1=list()
        col2=list()
        col3=list()
        for k in authors_start_end_injected.keys():
            col1.append(k)
            col2.append(authors_start_end_injected[k])
            col3.append(correct_start_end_injected[k])
        #
        df = pd.DataFrame({
            'id_istance': col1,
            'predicted linking': col2,
            'correct linking': col3
        })
        df.to_csv('result.csv', index=False, header=True)

        print("THE FOLLOWING INSTANCES MUST BE CHECKE MANUALLY")
        print("AUTHORS {}".format(str(errors_author)))
        print("CORRECT {}".format(str(errors_correct)))

        # NB: we decided to only use the configuration reported by the authors
        # create the dict with the start end predicted by the best model we found

        # this dict contains the method with start and end predicted by the random forest of the best model we found
        best_model_start_end=dict()