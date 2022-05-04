# Cross-Modal Cloze Task

We realase the datasets for our ACL-2022 Findings's paper: **Cross-Modal Cloze Task: A New Task to Brain-to-Word Decoding**. The fMRI60_CMC and fMRI180_CMC datasets can be found at the file folder `datasets`. 

**Format of data:**

* `word_stimuli.txt` contains the stimulus words that used in the fMRI experiments. 
* `sentences.txt` contains the stimulus word and its corresponding sentence, seperated by `|`. Format: word | sentence.
* `synonyms.txt` contains the stimulus word and its synonyms, seperated by a white space. Format: word syn1 syn2.
* `fMRI.txt` contains the link that realeased the fMRI data evoked by word stimuli, need to be preprocessed to 1 fMRI image per word per subject.


**To generate contexts for the stimulus words:**
```
python generate_contexts.py --data fMRI180
```

