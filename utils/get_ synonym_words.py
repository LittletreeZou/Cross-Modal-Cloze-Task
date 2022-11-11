'''
2021.11.1 Littletree

通过WordNet获取词语的同义词
Sci_word: 60 nouns
NC_word: 180 words (noun, verb, adj, adv)

ref:
https://www.tutorialspoint.com/how-to-get-synonyms-antonyms-from-nltk-wordnet-in-python
https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
'''
import re
from nltk.corpus import wordnet
from fmri_preprocessing import get_nc_fmri_words
from text_preprocessing import load_text, save_data


def get_wordnet_synonyms(word, pos):
    '''
    根据词性获取同义词, pos = noun, verb, adj, adv分别对应"n", "v", "a", "r"
    '''
    syn = set([])
    for synset in wordnet.synsets(word, pos):
        for lemma in synset.lemmas():
            syn.add(lemma.name())  # add the synonyms
    # post processing: remove words with link _ and -
    syn1 = []
    for w in list(syn):
        if (w == word) or ('-' in w) or ('_' in w):
            continue
        else:
            syn1.append(w)
    syn = [word] + syn1
    return syn



if __name__ == '__main__':

    DATA_FLAG = 'nc'

    # load data
    pos_dict = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}
    if DATA_FLAG == 'nc':
        fmri_words, pos_tag, _ = get_nc_fmri_words(subject='M15')
        # get pos tagging: 1: noun, 2: verb, 3: adjective, 4: adverb
        temp_dict = {1: "noun", 2: "verb", 3: "adjective", 4: "adverb"}
        pos_tag = [temp_dict[p] for p in pos_tag]
    else:
        fmri_words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
        pos_tag = ['noun' for _ in range(len(fmri_words))]

    syn_list = []
    for i, w in enumerate(fmri_words):
        syn = get_wordnet_synonyms(w, pos=pos_dict[pos_tag[i]])
        syn_list.append(' '.join(syn))

    if DATA_FLAG == 'nc':
        save_path = '/home/sxzou/concept_decoding/NC_data/text/wordnet_synonyms.txt'
    else:
        save_path = '/home/sxzou/concept_decoding/Science_data/wordnet_synonyms.txt'
    save_data(syn_list, save_path)