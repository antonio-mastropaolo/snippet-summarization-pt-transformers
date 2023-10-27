# Towards Summarizing Code Snippets Using Pre-Trained Transformers

Towards our goal of automatically summarizing code snippets, we first manually built a dataset featuring 6.6k comments that have been (i) classified based on their type (e.g, code summary, TODO), and (ii) linked to the exact code statements they document. Second, we used such a dataset to train a multi-task DL model taking as input a comment and being able to (i) classify whether it represents a ''code summary'' or not, and (ii) link it to the code statements it documents. Our trained model identifies code summaries with 84% accuracy and is able to link them to the documented lines of code with recall and precision higher than 80%. Third, we run this model on 10k Java open-source projects, automatically identifying code summaries and linking them to the related documented code. This allowed the building of a large-scale dataset of documented code snippets that has then been used to train a new DL model able to document code snippets automatically.

### Repository Structure:

  - <a href="https://github.com/snippet-summarization/icse23/tree/main/Code">Code</a> contains the code we developed for both parts of the approach. In other words, under <a href="https://github.com/snippet-summarization/icse23/tree/main/Code/Linking">Code/Linking</a> you can find the code concerning the automatic classification of code comments and likage to the documented code, while <a href="https://github.com/snippet-summarization/icse23/tree/main/Code/Code-Snippet-Summarizers/T5">Code/Code-Snippet-Summerizers</a> contains the code for training and testing **STUNT** (i.e, the T5-based model in charge of documenting code snippets). We also publicly release the code implementing the baseline, <a href="https://github.com/snippet-summarization/icse23/tree/main/Code/Code-Snippet-Summarizers/IR-Jaccard">Code/Code-Snippet-Summerizers/IR-Jaccard</a>.
  Finally, under <a href="https://github.com/snippet-summarization/icse23/tree/main/Code/Eval">Eval</a>, we release the code used to evaluate the models when summarizing a code snippet. To this extent, we re-use the *bleu.py* file provided by Huang *et al.* in
  <a href="https://github.com/huangshh/RLComGen/blob/master/bleu.py">Towards automatically generating block comments for code snippets</a>.


  - The Datasets :open_file_folder: are available at the following link: https://drive.google.com/drive/folders/1a8UOR5g9zSO0mPv293_xs0PSWLlQZ7BF?usp=sharing

    The GDrive folder contains three subfolders:
       * <a href="https://drive.google.com/drive/folders/10aqQAUqO1C3skNrBwxzeQJsFOxvlerbX?usp=sharing">*Manually-Annotated-Dataset*</a>, which contains the manually labeled dataset featuring 6.6k instances. Such a dataset is structured with the following columns:
          *  *javaClass*: The Java class containing the method of interest.
          *  *originalMethod*: The original Java method extracted from the beforementioned *javaClass*.
          *  *Comment*: The comment we manually labeled within the *originalMethod*.
          *  *spanComment*: The absolute (i.e, relative to the *javaClass*) byte position where the selected *Comment* begins and ends.
          *  *documentedCode*: The code snippet linked to the *Comment*.
          *  *category*: The selected category for the *Comment*.
          *  *isCodeSummary*: A boolean flag, indicating whether the category is a summary or not.
          *  *tokensLength*: The number of tokens featuring the *originalMethod*.
          *  *linkingInstance*: Preprocessed method containing the special tokens <start> <end> as well as <comment> </comment>. The former are used to mark the documentedCode, while the latter are used to highlight the selected comment*.
          *  *contiguous*: (c) for contiguous code selection, (nc) for non-contiguous code selection and (x) for instances without documented code.
          *  *preparedComment*: Contains the selected code comment for that given instance
          *  *inputClassificationTask*: Preprocessed input method for the classification task
          *  *targetClassification*: "CODE_SUMMARY" if the selection comment has been labelled as a code summary, "OTHER" otherwise.
          *  *inputLinkingTask*: Preprocessed input method for the linking task
          *  *targetLinkingTask*: Marked lines documenting the selected code. For example: <2> <3> <6>
       * <a href="https://drive.google.com/drive/folders/1DbJ4lOBhIg2PoW0O3xG1xx2gZUyIioVU?usp=sharing">*Large-Scale-Dataset*</a>, which contains the large scale dataset built using **SALOON**, the T5-based model we developed to automatically classify code comments by linking them to the documented piece of code.
       * <a href="https://drive.google.com/drive/folders/1j28g7xje4Qi20jSycWullE9RVzPv4-t-?usp=sharing">*T5 pre-training*</a>, which contains the dataset for pre-training the T5 model that we built starting from the <a href="https://github.com/github/CodeSearchNet">CodeSearchNet</a> dataset.


  - The trained models are available at the following link: https://drive.google.com/drive/folders/1ylGbBRsZ1ZN4H8MbPhudRrpe5KhN1IMR?usp=sharing
    
    The GDrive folder contains four subfolders containing the following models:
     * *Classification&Linking*, where you can find **SALOON**.
     * *CodeSnippetDocumentation*, containing **STUNT**, the model we develop to support the task of code snippet summarization.
     * *RLCom-T5-Replica*, where we make available the model trained using the dataset released in <a href="https://www.sciencedirect.com/science/article/pii/S0950584920301427?casa_token=jW82qRE6oDgAAAAA:Af44jxT9CVnaz7wdFu_KPJx--aawBaVmLtyFLXavZLirD5meTexlR6_gf-CdOMVZMhvWkdB54mY">Towards automatically generating block comments for code snippets</a>  
     * *T5-Pre-Trained-Model*, the T5 pre-trained model we use as the backbone for both approaches presented in our study, namely **SALOON** and **STUNT**.


  - The results :page_facing_up: of our study are available at the following link: https://drive.google.com/drive/folders/1gvkvqR5PJtW_WfrmIYa16nRVmfqVYvsI?usp=sharing
    
    The GDrive folder contains four subfolders, each reporting the predictions we obtained for each part of our approach.
    
    In details:
      * *Classification&Linking* contains the results of each model we use for such a task. Thus, we report the results achieved by the three different baselines (i.e, blank-line, random-forest, and token-based-text similarity) and **SALOON**.\
      NB: Concerning **SALOON**, we also release the results of the hyper-parameter tuning phase when both T5 training strategies are adopted, meaning with-pretraining and no-pretraining.
      * *Automated-Code-Snippet-Documentation*. In this folder, we make available the results achieved by **STUNT** on both the test and eval set (Hyper-parameters tuning).
      * *RLCom-Replication* contains the results achieved using a pre-trained transformer model (T5) on the dataset released in <a href="https://www.sciencedirect.com/science/article/pii/S0950584920301427?casa_token=jW82qRE6oDgAAAAA:Af44jxT9CVnaz7wdFu_KPJx--aawBaVmLtyFLXavZLirD5meTexlR6_gf-CdOMVZMhvWkdB54mY">Towards automatically generating block comments for code snippets</a>.


  - The SentencePiece Model we trained using the pre-training dataset so that the T5 model can deal with tokens belonging to a software-specific corpus is available at the following link: https://drive.google.com/drive/folders/14qSdyPaIjX_3XOykl3y3ejQ_c_AGKoli?usp=sharing
  

  - Ultimately, the folder <a href="https://github.com/snippet-summarization/icse23/tree/main/Misc">Misc</a> contains additional tables describing the exact configuration of hyper-parameters used when fine-tuning **SALOON** and **STUNT** and the results that have been achieved with the best hyper-parameters configuration.
