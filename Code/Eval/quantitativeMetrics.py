import nltk
from bleu import *
import sys
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer


def meteor(prediction,reference):
	return (nltk.translate.meteor_score.meteor_score([prediction], reference, gamma=0))

def rouge(prediction,reference,scorer):
	rougeL = scorer.score(prediction,reference)
	precision,recall,fmeasure = rougeL['rougeL'].precision, rougeL['rougeL'].recall, rougeL['rougeL'].fmeasure
	return precision,recall,fmeasure

def main():

	scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

	
	df_results = pd.read_csv('snippet-summarization-new-round/Results/Run-on-test/Summarizer/IR-Jaccard/IR-Jaccard.csv')
	targets = [item.rstrip('\n') for item in list(df_results['target'])]
	predictions = [item.rstrip('\n') for item in list(df_results['retrievedComment'])]
	
	

	#print(predictions[0])
	meteor_score = []
	sentence_bleu_4 = []
	overallPrecision = []
	overallRecall = []
	overallFmeasure = []
	perfect = []

	for target,prediction in tqdm(zip(targets,predictions)):
		
		meteor_score.append(meteor(prediction.split(), target.split()))
		p,r,f1 = rouge(prediction,target,scorer)
		sentence_bleu_4.append(score_sentence(prediction,target,4,1)[-1])

		overallPrecision.append(p)
		overallRecall.append(r)
		overallFmeasure.append(f1)
		
		if ''.join(target.split()) == ''.join(prediction.split()):
			perfect.append(True)
		else:
			perfect.append(False)

	# print("Perfect Predictions: {}/{}={}".format(perfect.count(True),len(targets),((perfect.count(True)/len(targets)) * 100)))
	# print("Meteor-Score: {}".format( (sum(meteor_score)/len(predictions)*100)))
	# print("BLEU-4: {}".format( (sum(sentence_bleu_4)/len(predictions)*100)))
	# print("Rouge-Precision: {}".format(sum(overallPrecision)/len(predictions)*100))
	# print("Rouge-Recall: {}".format( sum((overallRecall)/len(predictions))*100))
	# print("Rouge-Fmeasure: {}".format( (sum(overallFmeasure)/len(predictions))*100))

	df_results['meteor'] = meteor_score
	df_results['bleu4'] = sentence_bleu_4
	df_results['rougePrecision'] = overallPrecision
	df_results['rougeRecall'] = overallRecall
	df_results['rougeFmeasure'] = overallFmeasure
	df_results['perfect'] = perfect
	
	df_results.to_csv('IR-Jaccard-new.csv', index=False)


if __name__ == '__main__':
	main()
