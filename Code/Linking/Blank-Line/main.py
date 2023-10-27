import glob
from tqdm import tqdm
import os
import subprocess
from subprocess import Popen, PIPE
import re
import sys
import ast
import shutil
import pandas as pd
import json
from collections import Counter

class Process:
	stdout = None
	stderr = None
	cmd = ''
	SRCML_XPATH_ERROR = -2
	SRCML_PARSING_ERROR = -1

	def __init__(self, stdout=None, stderr=None, cmd=None):

		self.stderr = stderr
		self.stdout = stdout
		self.cmd = cmd

	def killProcess(self):
		self.process.kill()

	def runProcess(self, cmd=None, stdout=None, stderr=None, shell=None, encoding=None):

		try:
			self.process = subprocess.Popen(
				cmd if cmd is not None else self.cmd,
				stdout=subprocess.PIPE if stdout is None else stdout,
				stderr=subprocess.DEVNULL if stderr is None else stderr,
				shell=True if shell is None else shell,
				encoding=None if shell is None else encoding
			)

			stdout, stderr = self.process.communicate()
			return stdout, stderr

		except OSError:
			print("OSError: [Errno 7] Argument list too long: '/bin/sh'")
			return None, None


def retrieveLineNumber(process, offset, linkToFileJavaReadable):
	process.cmd = "perl -E '$off=shift;while(<>){$sum+=length;if($sum>=$off){say $.;exit}}' " + '{} {}'.format(offset, linkToFileJavaReadable)
	output,_ = process.runProcess()
	return int(re.findall(r'\d+',str(output))[0])

def endingComment(comment, method):
	match = re.search(re.escape(comment), method)
	if match != None:
		print("The String match at index % s, % s" % (match.start(), match.end()))
		return match.end()
	else:
		return -1

def matchCommentToCode(commentOffset, method):

	onlyValidMethodPortion = method.splitlines()[commentOffset:]

	# looking for the first valid line of code
	if onlyValidMethodPortion[0].strip() != "":
		lineStart = commentOffset+1
	else:
		return -1, -1


	# looking for the last valid line of code
	lineEnd = lineStart
	for line in onlyValidMethodPortion[1:]: #starting from 2, since the first item points to the beginning of the code selection
		if line.strip() == '':
			return lineStart, lineEnd
		else:
			lineEnd += 1

	return lineStart, lineEnd


def main():

	data = pd.read_csv('test-linking-classification.csv')
	process = Process()
	comments = {}
	javaFile = 'method.java'
	javaLineUnixFormat = 'method_unix.java'

	blankBaselineOutput = []
	blankBaselineOutputFlattened = []
	blankBaselineSpan = []
	linkedInstance = []
	blankBaselineMatched = []

	for(idx,row) in data.iterrows():

		if row['inputLinkingTask'] == "<NONE>":
			continue

		targetComments = eval(row['comment'])
		method = row['linkingInstance'].replace("<start>", '').replace("<end>", '')
		comments[idx] = []

		for comment in targetComments:
			for commentItemLine in comment.split('\n'):
				if commentItemLine.strip() != '':
					comments[idx].append(commentItemLine)

					if commentItemLine.strip().startswith('//'):
						parsedCommentLine = ' "{}" '.format(commentItemLine.strip().lstrip('//').strip())

					elif commentItemLine.strip().startswith('/*') and commentItemLine.strip().endswith('*/'):
						parsedCommentLine = ' "{}" '.format(commentItemLine.strip().lstrip('/*').rstrip('*/').strip())

					elif commentItemLine.strip().endswith('*/') and not commentItemLine.strip().startswith('/*'):
						parsedCommentLine = ' "{}" '.format(commentItemLine.strip().rstrip('*/').strip())

					else:
						parsedCommentLine = ' "{}" '.format(commentItemLine.strip())

					parsedCommentLine = '<comment> {} </comment>'.format(parsedCommentLine)


		end = endingComment(parsedCommentLine, method)

		with open(javaFile, 'w') as f:
			f.write(method)

		process.cmd = "perl -p -e 's/\r$//' < {} > {}".format(javaFile, javaLineUnixFormat)
		process.runProcess()
		lastCommentLine = retrieveLineNumber(process, end, javaLineUnixFormat)
		codeSpan = matchCommentToCode(lastCommentLine, method)
		methodLines = method.splitlines()
		methodLines[codeSpan[0]-1] = "<start> {}".format(methodLines[codeSpan[0]-1])
		methodLines[codeSpan[1]-1] = "{} <end>".format(methodLines[codeSpan[1]-1])

		# print("**********************\n")
		# print("Code Comment: {}".format('\n'.join(comments[idx])))
		# print("Code Span Lines: ", codeSpan)
		# print("\n".join(methodLines))
		# print("**********************\n")

		preparedTarget = ""
		for line in methodLines:
			preparedTarget += (line + ' <nl> ')
		preparedTarget = ' '.join(preparedTarget.split())


		blankBaselineOutputFlattened.append(preparedTarget)
		blankBaselineOutput.append('\n'.join(methodLines))
		blankBaselineSpan.append(codeSpan)
		
		linkedLines = ""
		for i in range(codeSpan[0],codeSpan[1]+1):
			linkedLines += "<{}> ".format(i)
		blankBaselineMatched.append(linkedLines.strip())
		linkedInstance.append(row['targetLinkingTask'])
	
	new_df = pd.DataFrame(zip(blankBaselineOutput, blankBaselineMatched, blankBaselineSpan, linkedInstance), columns=['blankBaselineOutput', 'blankBaselineMatched', 'blankBaselineSpan' ,'targetLinkingTask'])
	#flattening the oracle

	new_df.to_csv("results.csv")

if __name__ == '__main__':
	main()


