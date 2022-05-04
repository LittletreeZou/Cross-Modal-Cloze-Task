import codecs
import re


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


def load_sents(data_flag='nc'):
    '''
    load sentences corresponding to the word stimuli
    "nc" refers to data of Pereira 2018; 'sci' refers to data of Mitchell 2008.
    '''
    if data_flag == 'nc':
        datapath = "/home/sxzou/concept_decoding/NC_data/text/expt1_stim_sents.txt"
    else:
        datapath = "/home/sxzou/concept_decoding/Science_data/sentences.txt"

    readfile = codecs.open(datapath, 'r', 'utf-8')

    concepts = []
    sents = []

    for line in readfile:
        if data_flag == 'nc':
            line1 = line.strip().split('\t')
        else:
            line1 = line.strip().split('    ')  # 4个空格
        assert len(line1) == 2, 'error'
        word = line1[0].strip()
        sent = line1[1].strip()
        concepts.append(word)
        sents.append(sent)
    readfile.close()
    return concepts, sents


def find_mask_index(concept, split_sent):
    '''
    使用正则表达式去找concept出现在句子中的位置

    Note: re.match() checks for a match only at the beginning of the string
    '''
    index = -1
    for i, word in enumerate(split_sent):
        if re.match(concept, word) is not None:
            index = i
            break
    return index


  
def generate_mask_sentence(concepts, sents, mask_token='[MASK]'):
    '''
    mask the concept word in sentence
    '''
    mask_words = []
    mask_sents = []

    for i, sent in enumerate(sents):
        sent1 = sent.split(' ')
        index = find_mask_index(concepts[i], sent1)
        if index == -1:
            print(concepts[i], sent)
            mask_w = ''
            mask_s = sent
        else:
            # 处理有标点符号的情形
            if sent1[index][-1] in [',', '.', '?', '!']:
                mask_w = sent1[index][:-1]
                sent1[index] = mask_token + sent1[index][-1]
            # 处理特殊情况：后面加's
            elif sent1[index][-2:] == "'s":
                mask_w = sent1[index][:-2]
                sent1[index] = mask_token + sent1[index][-2:]
            # 处理特殊情况：law-imposed
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

    words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
    concepts, sents = load_sents(data_flag='sci')
    mask_words, mask_sents = generate_mask_sentence(concepts, sents, mask_token='[MASK]')




