import datetime
import codecs
import re


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


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


def save_data(data, data_path):
    f = codecs.open(data_path, 'w', 'utf-8')
    for text in data:
        f.write(text)
        f.write('\n')
    f.close()
    print(data_path, ' saved.')


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


def get_ref_for_word(tokenizer, data_flag='nc'):

    word2ref = {}
    if data_flag == 'nc':
        data_path = '/home/sxzou/concept_decoding/NC_data/text/human_corrected_synonyms.txt'
    else:
        data_path = '/home/sxzou/concept_decoding/Science_data/human_corrected_synonyms.txt'
    text = load_text(data_path)
    for line in text:
        ref = []
        words = line.split(' ')
        for w in words:
            ids = tokenizer.encode(w)
            if len(ids) > 3:
                ref.extend([tokenizer.decode([id]) for id in ids[1:-1]])
            else:
                ref.append(w)
        word2ref[words[0]] = list(set(ref))
    return word2ref


def generate_reference(concepts, mask_words, tokenizer, data_flag='nc'):
    '''
    the reference word could be word in concepts or in mask_words

    :param concepts: concept word corresponding to each sentence
    :param mask_words: the actual word masked in the sentence, may not be the same as in concepts
    '''
    word2ref = get_ref_for_word(tokenizer, data_flag)
    refs = []
    for i, w in enumerate(concepts):
        ref = word2ref[w]
        mask_w = mask_words[i]
        if mask_w != w:
            ids = tokenizer.encode(mask_w)
            if len(ids) > 3:
                ref.extend([tokenizer.decode([id]) for id in ids[1:-1]])
                ref = list(set(ref))
            else:
                ref.append(mask_w)
        refs.append(ref)
    return refs



if __name__ == '__main__':

    # words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
    import numpy as np
    #concepts, sents = load_sents(data_flag='sci')
    #mask_words, mask_sents = generate_mask_sentence(concepts, sents, mask_token='[MASK]')

    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #
    # # word2ref = get_ref_for_word(tokenizer, data_flag='sci')
    # # print(word2ref)
    # refs = generate_reference(concepts, mask_words, tokenizer, data_flag='sci')
    # for ref in refs:
    #     print(ref)
    # print(len(sents))
    # lens = []
    # for sent in sents:
    #     l = len(sent.split(' '))
    #     lens.append(l)
    #
    # lens = np.array(lens)
    # print(lens.min(), lens.max(), lens.mean(), lens.std())

    syn_count = 0   # 有同义词的词语数量
    syn_number = 0  # 同义词数量
    data_path = '/home/sxzou/concept_decoding/NC_data/text/human_corrected_synonyms.txt'
    #data_path = '/home/sxzou/concept_decoding/Science_data/human_corrected_synonyms.txt'
    text = load_text(data_path)
    for line in text:
        words = line.split(' ')
        if len(words) > 1:
            syn_count += 1
            syn_number += (len(words) - 1)
    print(syn_count, syn_number)
    print(syn_count/len(text), syn_number/syn_count)



