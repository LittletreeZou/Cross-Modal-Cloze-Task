# Cross-Modal Cloze Task

We realase the datasets and code for our ACL-2022 Findings's paper: [Cross-Modal Cloze Task: A New Task to Brain-to-Word Decoding](https://aclanthology.org/2022.findings-acl.54.pdf). 


## Datasets
The fMRI60_CMC and fMRI180_CMC datasets can be found at the file folder `datasets`. 

**Format of data:**

* `word_stimuli.txt` contains the stimulus words that used in the fMRI experiments. 
* `sentences.txt` contains the stimulus word and its corresponding sentence, seperated by `|`. Format: word | sentence.
* `synonyms.txt` contains the stimulus word and its synonyms, seperated by a white space. Format: word syn1 syn2.
* `fMRI.txt` contains the link that realeased the fMRI data evoked by word stimuli, need to be preprocessed to 1 fMRI image per word per subject.


**To generate contexts for the stimulus words:**
```
python utils/generate_contexts.py --data fMRI180
```

## Code

Sorry, the code is messy. The main steps for running the code are as follows:

**Step 1: Voxel selection**
We use the matlab code [trainVoxelwiseTargetPredictionModels.m](https://www.dropbox.com/s/l6hk9zkf2wvcflb/trainVoxelwiseTargetPredictionModels.m?dl=1) to compute the informative score of voxels for each subject.

**Step 2: Cross-modal mapping**
The file `cross_modal_mapping.py` contains all the code for training a regression model to map fMRI voxels to word embeddings.
 
**Step 3: Feature fusion and predict**
* The file `bert_baseline.py` corresponds to the baseline mentioned in our paper.
* The file `bert_fusion.py` corresponds to our proposed method.


Note: You need to change all paths for file reading and writing based on how you organize your data and code.
