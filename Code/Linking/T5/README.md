# How to Train SALOON


### STEP-1

*  ##### How to train a new <a href='https://github.com/google/sentencepiece/blob/master/python/README.md'>SentencePiece Model</a>

   Before training the [T5 small](https://github.com/google-research/text-to-text-transfer-transformer), namely the core of our approach (SALOON), it is important to also train a new tokenizer (sentencepiece model) on a software-specific corpora, such that the model can deal with the expanded vocabulary given by the Java programming language and Tenchnical natural language as well. 
   
   To train a new sentencepiece model, you can use the <a href="https://drive.google.com/file/d/1dPM4qQsSbR49QvykPdA02iC_9WEJ-Qzc/view?usp=sharing">pre-training instances</a> we collected from CodeSearchNet.

    ```
    pip install sentencepiece==0.1.96
    import sentencepiece as spm
    spm.SentencePieceTrainer.train('--input=pretraining-instances.txt --model_prefix=sp --vocab_size=32000 --bos_id=-1  --eos_id=1 --unk_id=2 --pad_id=0 --shuffle_input_sentence=true --character_coverage=1.0  --user_defined_symbols="<comment>,</comment>,<start>,<end>,<nl>"') 
    ```

    Our trained tokenizer is available here: https://drive.google.com/drive/folders/14qSdyPaIjX_3XOykl3y3ejQ_c_AGKoli?usp=sharing

### STEP-2
* ##### Setup a Google Cloud Storage (GCS) Bucket
    To setup a new GCS Bucket for training and fine-tuning a T5 Model, please follow the original guide provided by Google: Here the link: https://cloud.google.com/storage/docs/quickstart-console
    *NB: You need GCS to store the datasets and the model's checkpoints. Once you created your own GCS bucket, make sure to load the dataset for training and evaluating the model, then change the paths accordingly in the provided jupyter notebook*

### STEP-3
 * Once you have everything set up, you can run the provided jupyter notebook.
  *NB: SALOON is a multi-task model, meaning it is trained to support two tasks at the same time: (i) Code Comment Classification and (ii) Linkage of Code Comments to the Documented Code.* 





    

