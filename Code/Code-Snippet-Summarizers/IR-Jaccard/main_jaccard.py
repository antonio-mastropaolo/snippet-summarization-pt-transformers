import pandas as pd
from tqdm import tqdm
import sys

def main():

    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('Splits/test{}.csv'.format(sys.argv[1]))
    
    #df_train = df_train.iloc[0:50]
    #df_test = df_test.iloc[0:50]

    best_matching_snippet = []
    retrievedCommentItem = []
    jaccardValue = []

    for idx_test,row_test in tqdm(df_test.iterrows()):
        targetCode = row_test['codeSnippet'].strip()
        codeSnippetTest_deduplicated = set(targetCode)
        jaccard = 0
        for idx_train, row_train in df_train.iterrows():

            trainCode = row_train['codeSnippet'].strip()
            codeSnippetTrain_deduplicated = set(trainCode)

            numerator = len(codeSnippetTest_deduplicated.intersection(codeSnippetTrain_deduplicated))
            denominator = len(codeSnippetTest_deduplicated.union(codeSnippetTrain_deduplicated))

            new_jaccard = round( (numerator / denominator), 2)
            if new_jaccard > jaccard:
                jaccard = new_jaccard
                retrievedComment = row_train['comment']
                matchingSnippet = trainCode

        best_matching_snippet.append(matchingSnippet)
        retrievedCommentItem.append(retrievedComment)
        jaccardValue.append(jaccard)
        # print("\n**********************************")
        # print("Jaccad Value: {}".format(jaccard))
        # print("Retrieved Comment: " + retrievedComment)
        # print("Code Snippet under Test: "+targetCode)
        # print("************************************\n")
    
    df_test['bestMatchingSnippet'] = best_matching_snippet
    df_test['jaccardValue'] = jaccardValue
    df_test['retrievedComment'] = retrievedCommentItem
    df_test.to_csv('test-results-{}.csv'.format(sys.argv[1]))

if __name__ == '__main__':
    main()


