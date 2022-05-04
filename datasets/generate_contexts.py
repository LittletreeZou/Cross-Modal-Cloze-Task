import codecs
import re
import os
import argparse


def load_text(data_path, lower=False):
    readfile = codecs.open(data_path, 'r', 'utf-8')
    corpus = []
    if lower:
        for line in readfile:
            sent = line.strip().split(' ')
            sent = ' '.join([w.lower() for w in sent])
            corpus.append(sent)
    else:
        for line in readfile:
            sent = line.strip()
            corpus.append(sent)
    readfile.close()
    return corpus


def load_sents(data_flag, data_file_path):
    '''
    Load stimulus word and its corresponding sentences.

    :param data_flag: which fMRI dataset to process, fMRI60 or fMRI180.
    :param data_file_path: the file directory of `datasets`.
    :return: (concepts, sents), both of which are list of str
    '''
    datapath = os.path.join(data_file_path, data_flag + "_CMC/sentences.txt")
    readfile = codecs.open(datapath, 'r', 'utf-8')

    concepts = []   # stimulus words
    sents = []

    for line in readfile:
        line1 = line.strip().split('|')
        assert len(line1) == 2, 'error'
        word = line1[0].strip()
        sent = line1[1].strip()
        concepts.append(word)
        sents.append(sent)
    readfile.close()
    return concepts, sents


def find_mask_index(concept, split_sent):
    '''
    Use regular expression to find the index of the stimulus word in the sentence.

    :param concept: a stimulus word
    :param split_sent: a sentence that has been split into word list
    :return: the index of the stimulus word in the sentences.
    '''
    index = -1
    for i, word in enumerate(split_sent):
        if re.match(concept, word) is not None:
            index = i
            break
    return index


def generate_mask_sentence(concepts, sents, mask_token='[MASK]'):
    '''
    Mask the concept word in the sentence to generate a context for the word.

    :param concepts: word stimuli, list
    :param sents: sentences, list
    :param mask_token: used as a placeholder, it should be the same for the pre-trained model you used.
    :return: (mask_words, mask_sents), both of which are list of str
    '''
    mask_words = []  # store the exact word that is replaced by the mask_token
    mask_sents = []  # store the masked sentence (context)

    for i, sent in enumerate(sents):
        sent1 = sent.split(' ')
        index = find_mask_index(concepts[i], sent1)
        if index == -1:
            print(concepts[i], sent)
            mask_w = ''
            mask_s = sent
        else:
            # to handle the punctuations
            if sent1[index][-1] in [',', '.', '?', '!']:
                mask_w = sent1[index][:-1]
                sent1[index] = mask_token + sent1[index][-1]
            # to handle the case where the concept word looks like word's
            elif sent1[index][-2:] == "'s":
                mask_w = sent1[index][:-2]
                sent1[index] = mask_token + sent1[index][-2:]
            # to handle a special word 'law-imposed'
            elif sent1[index] == 'law-imposed':
                mask_w = 'law'
                sent1[index] = mask_token + "-imposed"
            else:
                mask_w = sent1[index]
                sent1[index] = mask_token
            mask_s = ' '.join(sent1)
        mask_words.append(mask_w)
        mask_sents.append(mask_s)
    return mask_words, mask_sents



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='fMRI60', help='fMRI60 or fMRI180')
    args = parser.parse_args()

    concepts, sents = load_sents(data_flag=args.data, data_file_path='./datasets')
    _, contexts = generate_mask_sentence(concepts, sents, mask_token='[MASK]')
    print('Context example: ', contexts[0])
    
    
