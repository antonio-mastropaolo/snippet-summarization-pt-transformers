import sys
import os
import csv
import xml

from utils.code_feature_extraction import CodeFeatureExtraction
from utils.statement_utilities import StatementUtilities

import pandas as pd

# input_folder = "/Users/matteo.ciniselli/OneDrive - USI/code/007_Comment_Labeling/inp/1_baseline/dataset"

class CodeFeatures:
    def __init__(self, separator, set, xml_folder):
        self.header=list()
        self.separator=separator
        self.set=set

        self.header.extend(["id", "start", "end", "code"])

        self.header.extend(["if", "while", "for", "enhancedfor", "try", "decl", "return", "break", "throw"])
        self.header.extend(["num substatement", "nested level", "number lines", "same method", "same variable", "previous blank", "next blank"])

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


    def return_statement_types(self, statement):

        """
        Nine common types of statements including If-statements, While-statements, For-statements, EnhanceFor-statements,
        TryCatch-statements, Variable Declaration-statements, Return-statements, Break-statements, Throw-statements
        """

        types=["if", "while", "for", "enhancedfor", "try", "decl", "return", "break", "throw"]

        values=list()
        for t in types:
            if statement.type==t:
                values.append("1")
            else:
                values.append("0")

        return values

    def compute_features(self, id_instance, statement, has_same_methods, has_same_variables, previous_blank, next_blank):
        result=list()
        result.append(str(id_instance))
        result.append(str(statement.start_line))
        result.append(str(statement.end_line))
        result.append(statement.code)
        result.extend(self.return_statement_types(statement))
        result.append(str(statement.num_substatements))
        result.append(str(statement.nested_level))
        result.append(str(statement.num_lines))
        result.append(str(has_same_methods))
        result.append(str(has_same_variables))
        result.append(str(previous_blank))
        result.append(str(next_blank))

        return self.separator.join(result)

    def is_statement_commented(self, statement, commented_lines):

        for el in range(statement.start_line, statement.end_line+1):
            if el not in commented_lines:
                return 0

        return 1



    def process_files(self):

        for file in self.xml_files:
            if self.set not in file:
                continue

            # lines=self.read_data(os.path.join(self.xml_folder,file))

            lines=pd.read_csv(os.path.join(self.xml_folder,file))

            errors=0

            features=dict()

            is_statement_commented_dict=dict()

            print("CODE FEATURES {} LINES".format(len(lines.index)))

            for index, line in lines.iterrows():
                try:
                    id_instance=int(line["id"])

                    print(id_instance)

                    xml_code=line["xml"]
                    xml_code="\n".join(eval(xml_code))

                    commented_lines=eval(line["referenced_lines"])

                    statements=dict()

                    su=StatementUtilities()
                    cf=CodeFeatureExtraction()

                    statements = su.return_statements(statements, xml_code)


                    method_dict, variable_dict= cf.return_methods_and_variables(xml_code)
                    # print(method_dict)
                    # print(variable_dict)

                    for key in statements.keys():
                        if statements[key].type=="EMPTY":
                            continue

                        key_feature="{}_{}".format(id_instance, statements[key].start_line)

                        # statements[key].print_statement()
                        has_same_methods, has_same_variables=cf.find_same_methods_variable_close_statement(statements, key, method_dict, variable_dict)

                        previous_blank, next_blank=cf.find_blank_lines_close_statements(statements, key)

                        feature=self.compute_features(key_feature, statements[key], has_same_methods, has_same_variables, previous_blank, next_blank)
                        features[key_feature]=feature

                        res=self.is_statement_commented(statements[key], commented_lines)
                        is_statement_commented_dict[key_feature]=res


                except Exception as e:
                    errors+=1
                    print("ERROR__ERROR")
                    print(e)


            print("{} ERRORS".format(errors))
            return self.header, features, is_statement_commented_dict