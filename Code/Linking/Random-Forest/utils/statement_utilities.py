import xml.etree.ElementTree as ET
import string

import xml.etree.ElementTree as ET
from utils.statement_helper import Statement

class StatementUtilities:
    def __init__(self):
        pass

    def get_line(self, elem):
        for k in elem.attrib.keys():
            if "start" in k:
                pos = elem.attrib[k]

                return int(pos.split(":")[0])

    def get_text(self, elem):

        if elem is None:
            return "__NO_CODE__"

        text = ET.tostring(elem, encoding='utf8', method='text').decode("utf-8")

        lines = text.split("\n")
        start_line = 0
        end_line = 0
        for i, l in enumerate(lines):
            if len(l.strip()) > 0:
                start_line = i
                break

        for i in range(len(lines) - 1, -1, -1):
            if len(lines[i].strip()) > 0:
                end_line = i
                break

        return "\n".join(lines[start_line:end_line + 1])

    def add_class(self, elem):
        '''
        given an elem, returns if it is one of the types identified by the authors of the paper
        '''
        tag = elem.tag
        # print(tag)
        pos = self.get_line(elem)

        if pos is None:
            return None

        s = None

        if "if_stmt" in tag:
            s = Statement("if", pos, elem, self.get_text(elem))
        elif "comment" in tag:
            s = Statement("comment", pos, elem, self.get_text(elem))
        elif "return" in tag:
            s = Statement("return", pos, elem, self.get_text(elem))
        elif "do" in tag or "while" in tag:
            s = Statement("while", pos, elem, self.get_text(elem))
        elif "break" in tag:
            s = Statement("break", pos, elem, self.get_text(elem))
        elif "for" in tag:

            type = "for"
            for sub_elem in elem.iter():
                if "control" in sub_elem.tag:
                    if ":" in self.get_text(sub_elem):
                        type = "enhancedfor"

            s = Statement(type, pos, elem, self.get_text(elem))
        elif "decl" in tag:
            s = Statement("decl", pos, elem, self.get_text(elem))
        elif "throw" in tag:
            s = Statement("throw", pos, elem, self.get_text(elem))

        elif "try" in tag:
            s = Statement("try", pos, elem, self.get_text(elem))

        return s

    def return_ok_lines(self, statements):
        '''
        return the lines that are OK (i.e., we don't need to add them to a statement
        '''
        ok_lines = list()
        for k in statements.keys():
            statement_curr = statements[k]
            ok_lines.append(statement_curr.start_line)
            # if we split the declaration statement on more than one line, we ignore the other lines
            if statement_curr.type in ["return", "break", "decl", "throw"]:
                if statement_curr.num_lines > 1:
                    for i in range(statement_curr.start_line + 1, statement_curr.end_line + 1):
                        ok_lines.append(i)

        # print(ok_lines)
        return ok_lines

    @staticmethod
    def count_substatements(statements, curr_position):
        num_substatements = 0
        start_line = statements[curr_position].start_line
        end_line = statements[curr_position].end_line

        for k in statements.keys():
            # ignore the curr statement
            if k == curr_position:
                continue
            statement_curr = statements[k]
            # we did not count the empty statement as statement
            if statement_curr.type == "EMPTY":
                continue

            if statement_curr.start_line >= start_line and statement_curr.end_line <= end_line:
                num_substatements += 1

        return num_substatements


    def return_statements(self, statements, code):

        tree = ET.fromstring(code)

        code=self.get_text((tree))

        # print("CODE")
        # print(code)

        lines=code.split("\n")

        for i,l in enumerate(lines):
            if len(l.strip())==0:
                statement=Statement("EMPTY", i+1, None, "")
                statements[i+1]=statement

        for elem in tree.iter():
            # print(elem.tag)

            res = self.add_class(elem)
            if res is not None:
                # we want to be sure that the elements we're finding are not part of a line
                # because for example int=0 in a for condition is considered as a declaration
                # we ignore throw type
                start_line=res.start_line
                code=self.get_text(res.elem)
                first_line=code.split("\n")[0]

                if first_line.replace(" ","")!=lines[start_line-1].replace(" ",""):
                    if res.type !="throw":
                        continue

                statements[res.start_line] = res

            # print((elem.attrib.keys()))
            # print(elem.text)

        # lines_ok contain the lines that have been included in a statement
        # if we have a while, we only insert the first line (since the nested line must have another type)
        lines_ok=self.return_ok_lines(statements)

        for i,l in enumerate(lines):
            if i+1 not in lines_ok:
                statement_curr=Statement("OTHER", i+1, None, l)
                statements[i+1]=statement_curr

        statements=dict(sorted(statements.items()))

        # remove other lines that involve more than one line
        # e.g., a log message on 3 different lines, we only need to add a single statement with three lines
        # rather than 3 statements of a single line

        keys_to_delete=list()

        for k in statements.keys():
            statement_curr=statements[k]
            if statement_curr.type != "OTHER":
                continue

            if k in keys_to_delete:
                continue

            last_char=lines[k-1].strip()[-1]
            if last_char not in [")", ";", "{", "}"]:
                num_lines=1
                for i in range(k, len(lines)):
                    # print("CHECK LINE {}".format(i))

                    if i in statements.keys():
                        statement_=statements[i]
                        # print(statement_.type)
                        if statement_.type != "OTHER":
                            break

                        else:
                            keys_to_delete.append(i+1)

                    if len(lines[i].strip()) ==0:
                        break

                    last_char = lines[i].strip()[-1]

                    num_lines += 1
                    statement_curr.code+="\n{}".format(lines[i])
                    # print("NEW STATEMENT")
                    # print(statement_curr.code)
                    if last_char in [")", ";", "{", "}"]:
                        break

                statement_curr.add_num_lines_and_end(num_lines, statement_curr.start_line+num_lines-1)

        for k in keys_to_delete:
            if statements[k].type =="EMPTY" or statements[k].type=="comment":
                continue
            del statements[k]

        # print(keys_to_delete)

        # remove lines that involve only punctuation

        keys_to_delete=list()
        for k in statements.keys():
            statement_curr=statements[k]
            if statement_curr.type != "OTHER":
                continue

            curr_code=statement_curr.code
            for character in string.punctuation:
                curr_code = curr_code.replace(character, '')

            if len(curr_code.strip())==0:
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del statements[k]


        statements=dict(sorted(statements.items()))

        # for each statement we add the number of substatements

        for k in statements.keys():
            statement_curr=statements[k]
            num_substatement=StatementUtilities.count_substatements(statements,k)
            statement_curr.add_substatement(num_substatement)

        return statements