import javalang
from javalang.tree import *

import xml.etree.ElementTree as ET

class CodeFeatureExtraction():
    def __init__(self):
        pass

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

    def get_previous_statement(self, statements, curr_position, ignore_blank=True):
        '''
        find the previous statement
        if we set ignore_blank=False we return the previous statement also if it is empty
        otherwise we say that there is no previous statement
        '''
        curr_statement = statements[curr_position]
        if curr_statement.type == "EMPTY":
            return None
        # for example } else {
        if curr_statement.code.strip()[0] == "}":
            return None

        previous_position = curr_position - 1

        # Finally, we checked whether there were blank lines before and after a statement.
        # A professional programmer with proper pro- gramming style would use individual blank lines
        # within the func- tions for separating the critical code sections to increase the func- tionality
        # of the software (Vermeulen, 2000). Therefore, blank lines
        # usually represent the ends of the scopes of comments.
        if previous_position in statements.keys():
            if statements[previous_position].type == "EMPTY":
                if ignore_blank:
                    return None
                else:
                    return statements[previous_position]

        for key in statements.keys():
            statement_curr = statements[key]

            if statement_curr.end_line == previous_position:
                # if the next statement ends with a bracket we're changing the level of indentation
                if statement_curr.code.strip()[-1] == "{":
                    return None
                return statement_curr

        return None

    def get_next_statement(self, statements, curr_position, ignore_blank=True):
        '''
        find the next statement
        if we set ignore_blank=False we return the next statement also if it is empty
        otherwise we say that there is no next statement
        '''
        curr_statement = statements[curr_position]

        if curr_statement.type == "EMPTY":
            return None
        # for example } else {
        if curr_statement.code.strip()[-1] == "{":
            return None

        # we look for a statement that starts in the line just after the end of the current statement
        next_position = statements[curr_position].end_line + 1

        # Finally, we checked whether there were blank lines before and after a statement.
        # A professional programmer with proper pro- gramming style would use individual blank lines
        # within the func- tions for separating the critical code sections to increase the func- tionality
        # of the software (Vermeulen, 2000). Therefore, blank lines
        # usually represent the ends of the scopes of comments.
        if next_position in statements.keys():
            if statements[next_position].type == "EMPTY":
                if ignore_blank:
                    return None
                else:
                    return statements[next_position]

        for key in statements.keys():
            statement_curr = statements[key]

            if statement_curr.start_line == next_position:
                # if the next statement starts with a bracket we're changing the level of indentation
                if statement_curr.code.strip()[0] == "}" or statement_curr.code.strip()[0] == "{":
                    return None
                return statement_curr

        return None

    def return_methods_and_variables(self, code):

        tree = ET.fromstring(code)

        code = self.get_text((tree))

        code = "class ParseC {{ \n {} \n }}".format(code)
        # lines = code.split("\n")
        # print("CODE")
        # print(code)

        tree = javalang.parse.parse(code)

        tokens = javalang.tokenizer.tokenize(code)

        # we save all the identifiers in each line
        # in this way we can check in which line are the variables
        # since javalang is not able to report the position of a variable
        tokens_line = dict()
        for t in tokens:
            if "Identifier" not in t.__str__():
                continue
            token_value = t.value
            token_position = t.position.line - 1  # we need to remove the class line declaration

            if token_position not in tokens_line:
                tokens_line[token_position] = list()

            tokens_line[token_position].append(token_value)
            # print(dir(t))

        classes = [n for _, n in tree.filter(ClassDeclaration)]

        method_line = dict()
        variable_line = dict()

        for clazz in classes:

            for node in clazz.methods:

                # print(node.position)

                for _, invocation in node.filter(MethodInvocation):
                    position = invocation.position.line - 1
                    if position not in method_line.keys():
                        method_line[position] = list()

                    method_line[position].append(invocation.member)

                for _, declaration in node.filter(VariableDeclarator):
                    variable_name = declaration.name

                    for k in tokens_line.keys():
                        tokens = tokens_line[k]
                        if variable_name in tokens:
                            if k not in variable_line.keys():
                                variable_line[k] = list()

                            variable_line[k].append(variable_name)

        return method_line, variable_line

    def find_methods(self, statement, method_dict):
        # find all the methods in the statement
        curr_methods = dict()
        for x in range(statement.start_line, statement.end_line + 1):
            if x in method_dict.keys():
                for method_call in method_dict[x]:
                    curr_methods[method_call] = 1

        return curr_methods

    def find_variables(self, statement, variable_dict):
        # find all the variable in that statement
        curr_variables = dict()
        for x in range(statement.start_line, statement.end_line + 1):
            if x in variable_dict.keys():
                for variable_decl in variable_dict[x]:
                    curr_variables[variable_decl] = 1

        return curr_variables


    def find_same_methods_variable_close_statement(self, statements, curr_position, method_dict, variable_dict):
        '''
        return if the previous and next statement have at least one method call in common
        and if they have at least a variable name in common
        '''

        previous_statement=self.get_previous_statement(statements, curr_position)
        next_statement=self.get_next_statement(statements, curr_position)

        curr_statement=statements[curr_position]

        has_same_method=0
        has_same_variables=0

        if previous_statement is None and next_statement is None:
            return 0,0

        # methods present in the current statement
        curr_methods=self.find_methods(curr_statement, method_dict)

        # methods present in the current statement
        curr_variables=self.find_methods(curr_statement, variable_dict)

        if previous_statement is not None:
            previous_methods=self.find_methods(previous_statement, method_dict)
            previous_variables=self.find_variables(previous_statement, variable_dict)

            # curr_statement.print_statement()
            # previous_statement.print_statement()

            for k in previous_methods.keys():
                if k in curr_methods.keys():
                    has_same_method=1

            for k in previous_variables.keys():
                if k in curr_variables.keys():
                    has_same_variables = 1

        if next_statement is not None:
            next_methods=self.find_methods(next_statement, method_dict)
            next_variables=self.find_variables(next_statement, variable_dict)

            # curr_statement.print_statement()
            # next_statement.print_statement()

            for k in next_methods.keys():
                if k in curr_methods.keys():
                    has_same_method=1

            for k in next_variables.keys():
                if k in curr_variables.keys():
                    has_same_variables = 1

        return has_same_method, has_same_variables

    def find_blank_lines_close_statements(self, statements, curr_position):
        '''
        return if the previous and next statement are a blank line
        '''

        previous_statement = self.get_previous_statement(statements, curr_position, False)

        next_statement = self.get_next_statement(statements, curr_position, False)

        curr_statement = statements[curr_position]

        previous_blank=0
        next_blank=0

        if previous_statement is None and next_statement is None:
            return 0, 0

        if previous_statement is not None:
            # previous_statement.print_statement()
            # curr_statement.print_statement()
            if previous_statement.type=="EMPTY":
                previous_blank=1

        if next_statement is not None:
            if next_statement.type=="EMPTY":
                next_blank=1

        return previous_blank, next_blank