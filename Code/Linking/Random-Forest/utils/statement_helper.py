import xml.etree.ElementTree as ET

class Statement():
    def __init__(self, type, line, elem, code):
        self.type = type
        self.start_line = line
        self.elem = elem
        self.code=code

        self.num_lines=0
        self.nested_level=0
        self.end_line=0
        self.num_substatements=0

        self.compute_metric()

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

    def compute_metric(self):
        if self.elem is None:
            self.end_line=self.start_line
            self.num_lines=1
            self.nested_level=0
        else:
            code=self.get_text(self.elem)
            lines=code.split("\n")
            # num_brackets=code.count("{")
            self.num_lines=len(lines)
            self.nested_level=self.count_nested_level()
            self.end_line=self.start_line+self.num_lines-1

    def count_nested_level(self):
        code=self.code
        nested_level=0
        nested_level_curr=0
        for c in code:
            if c=="{":
                nested_level_curr+=1
            elif c=="}":
                nested_level_curr-=1

            if nested_level_curr>nested_level:
                nested_level=nested_level_curr

        return nested_level


    def add_num_lines_and_end(self, num_lines, end_line):
        self.num_lines=num_lines
        self.end_line=end_line

    def add_substatement(self, num_substatement):
        self.num_substatements=num_substatement

    def print_statement(self):

        print("{} type".format(self.type))
        print("START AT LINE {}, END AT LINE {} ({} LINES)".format(self.start_line, self.end_line, self.num_lines))
        print("LEVEL {}".format(self.nested_level))
        print("NUM SUBSTATEMENT {}".format(self.num_substatements))
        # if self.elem is not None:
        #     print("___")
        #     print(get_text(self.elem))
        print("___")
        print(self.code)
        print("________________")


    def print_statement_short(self):

        print("{} FROM {} TO {}".format(self.type, self.start_line, self.end_line))