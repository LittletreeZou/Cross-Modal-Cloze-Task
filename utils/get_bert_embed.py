'''
获取词语的bert embedding：embedding层 + 12层的每一层

注意bert采用bpe编码

STEP 1: 对每个词语的6个句子，获取句子13层的词向量表示，13 x (batch_size, seq_len, dim)
STEP 2: 从句子中去匹配我们的目标词语，对6个词向量取平均后作为我们的词语的词向量

'''
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel

from fmri_preprocessing import get_nc_fmri_words
from text_preprocessing import load_text, load_sents, generate_mask_sentence
from data_util import sid_to_wid



def get_glove_embed(words):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/sxzou/concept_decoding/data/glove_model.txt')
    embed = np.zeros((len(words), 300))
    for i, w in enumerate(words):
        try:
            embed[i] = model[w]
        except KeyError:
            print(i,w)
    return embed


def get_bert_embed_for_single_concept_sent(concept, sent, model, tokenizer, layer):

    input_ids = tokenizer.encode(sent, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids.cuda())
        hidden_states = outputs[2]  # 13 tuples of shape (batch_size, seq_len, hidden_size)
        sent_embed = hidden_states[layer].squeeze(0).cpu().numpy()    # (seq_len, hidden_size)

    # 取出concept的词向量
    input_ids = tokenizer.encode(sent)
    concept_embed = None
    answer_ids = None
    word_ids = tokenizer.encode(concept)[1:-1]
    n = len(word_ids)
    for i in range(len(input_ids)):
        if input_ids[i: i+n] == word_ids:
            concept_embed = sent_embed[i: i+n].mean(0)
            answer_ids = word_ids
    if concept_embed is None:
        print(concept, sent)
        print(word_ids, input_ids)
    return concept_embed, answer_ids


def get_bert_embed(fmri_words, concepts, mask_words, sentences, model, tokenizer, layer, data_flag='nc'):
    '''
    :param data_flag: "sci" means science data, "nc" means nature communication data
    :return:
    '''
    if data_flag == 'nc':
        word_vecs = np.zeros((180, 768))
        count = np.zeros(180)
        sid2wid = sid_to_wid(concepts, fmri_words)
        for i, sent in enumerate(sentences):
            embed, _ = get_bert_embed_for_single_concept_sent(mask_words[i], sent, model, tokenizer, layer)
            ind = sid2wid[i]
            word_vecs[ind] += embed
            count[ind] += 1
        assert (count == 6).sum() == 180, 'sentence number is wrong!'
        word_vecs = word_vecs / 6
    else:
        word_vecs = np.zeros((60, 768))
        for i, sent in enumerate(sentences):
            embed, _ = get_bert_embed_for_single_concept_sent(mask_words[i], sent, model, tokenizer, layer)
            ind = i//6
            word_vecs[ind] += embed
        word_vecs = word_vecs / 6
    return word_vecs


def get_word_embed_matrix(model):
    for layer in model.named_modules():
        if layer[0] == 'embeddings.word_embeddings':
            embed_matrix = layer[1].weight.data   # of shape (30522, 768)
    # convert matrix to numpy array of datatype np.float32
    embed_matrix = embed_matrix.cpu().numpy().astype(np.float32)
    return embed_matrix


def get_word_embed(words, embed_matrix, tokenizer):
    word_vecs = np.zeros((len(words), 768))
    for i, word in enumerate(words):
        word_id = tokenizer.encode(word)[1:-1]
        for id in word_id:
            word_vecs[i] += embed_matrix[id]
        word_vecs[i] = word_vecs[i] / len(word_id)
    return word_vecs



if __name__ == '__main__':

    # from text_preprocessing import load_text
    # words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
    # embed = get_glove_embed(words)
    # save_path = '/home/sxzou/concept_decoding/Science_data/glove_embed.npy'
    # np.save(save_path,  embed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    DATA_FLAG = 'sci'

    print('load model...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.cuda()
    model.eval()

    if DATA_FLAG == 'nc':
        fmri_words = get_nc_fmri_words(subject='M15')[0]
    else:
        fmri_words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')

    concepts, sents = load_sents(data_flag=DATA_FLAG)
    mask_words, _ = generate_mask_sentence(concepts, sents, mask_token='[MASK]')

    # get layer 0 embedding
    embed_matrix = get_word_embed_matrix(model)  # (30522, 768)
    word_vecs = get_word_embed(fmri_words, embed_matrix, tokenizer)
    save_path = '/home/sxzou/concept_decoding/data/wordvecs/sci/bert_embed_60.npy'
    np.save(save_path, word_vecs)

    # get layer 1-12 embedding
    layeravg = np.zeros((len(fmri_words), 768))
    for layer in range(7, 13):
        word_vecs = get_bert_embed(fmri_words, concepts, mask_words, sents, model, tokenizer, layer, data_flag=DATA_FLAG)
        layeravg += word_vecs
    save_path = '/home/sxzou/concept_decoding/data/wordvecs/sci/bert_layer12_60.npy'
    np.save(save_path, word_vecs)
    save_path = '/home/sxzou/concept_decoding/data/wordvecs/sci/bert_layeravg_60.npy'
    np.save(save_path, layeravg/6)


    # 统计词语的token长度
    # token_len = []
    # for word in fmri_words:
    #     token_len.append(len(tokenizer.encode(word)[1:-1]))
    # token_len = np.array(token_len)
    # print((token_len>=2).sum())
    # index = [i for i in range(180) if token_len[i]>1]
    # print([fmri_words[i] for i in index])

    # long_words = ['argumentatively', 'cockroach', 'deceive', 'sew']
    # for word in long_words:
    #     print(word)
    #     print([tokenizer.decode([i]) for i in tokenizer.encode(word)[1:-1]])





