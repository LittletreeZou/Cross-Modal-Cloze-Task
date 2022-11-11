import pandas as pd
from transformers import BertTokenizer
from text_preprocessing import load_sents, load_text


def tokenization(tokenizer, text_sent):
    '''
    :param tokenizer: bert tokenizer
    :param text_sent: list of sentences
    '''
    tokenized_text = []
    for sent in text_sent:
        token_ids = tokenizer(sent, return_tensors='np')['input_ids'].squeeze()
        token_ids = [int(x) for x in list(token_ids)]
        tokenized_text.append(token_ids)
    return tokenized_text


def get_local_bert_vocab(tokenizer, tokenized_text):
    '''
    don't take the bert's special token into account
    '''
    vocab = {}      # word: tf
    word2id = {}    # word: token id

    unk_num = 0
    total_num = 0

    special_ids = [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id]
    unk_id = tokenizer.unk_token_id
    for token_ids in tokenized_text:
        for ind in token_ids:
            if ind in special_ids:
                continue
            if ind == unk_id:
                unk_num += 1
                continue
            token = tokenizer.decode([ind])
            try:
                vocab[token] += 1
            except KeyError:
                vocab[token] = 1
                word2id[token] = ind
            total_num += 1
    return vocab, word2id, unk_num, total_num


def save_local_vocab(vocab, word2id, ref_word2id):
    word_list = []
    id_list = []
    tf_list = []

    for word, tf in vocab.items():
        word_list.append(word)
        id_list.append(word2id[word])
        tf_list.append(tf)

    for w, id in ref_word2id.items():
        if w not in word_list:
            word_list.append(w)
            id_list.append(id)
            tf_list.append(1)

    vocab_df = pd.DataFrame({'word': word_list, 'id': id_list, 'tf': tf_list})
    vocab_df.sort_values(by='tf', ascending=False, inplace=True)
    # save vocab
    vocab_df.to_csv('/home/sxzou/concept_decoding/data/local_bert_vocab_nc_sci_ref.csv', index=False)
    print('file saved!')
    return vocab_df


def check_out_multitoken_words(tokenizer, data_flag='nc'):
    '''
    看看有多少词语是多个token的
    '''
    if data_flag == 'nc':
        data_path = '/home/sxzou/concept_decoding/NC_data/text/human_corrected_synonyms.txt'
    else:
        data_path = '/home/sxzou/concept_decoding/Science_data/human_corrected_synonyms.txt'
    text = load_text(data_path)

    for line in text:
        words = line.split(' ')  #list
        for w in words:
            ids = tokenizer.encode(w)
            if len(ids) > 3:
                print(words[0], w, [tokenizer.decode([id]) for id in ids[1:-1]])


def get_refs_ids(tokenizer, nc_concepts, sci_concepts):

    mask_words, _ = generate_mask_sentence(nc_concepts, nc_sents, mask_token='[MASK]')
    nc_refs = generate_reference(nc_concepts, mask_words, tokenizer, data_flag='nc')

    mask_words, _ = generate_mask_sentence(sci_concepts, sci_sents, mask_token='[MASK]')
    sci_refs = generate_reference(sci_concepts, mask_words, tokenizer, data_flag='sci')

    refs = []
    for ref in nc_refs:
        refs.extend(ref)
    for ref in sci_refs:
        refs.extend(ref)
    refs = list(set(refs))

    word2id = {}
    for w in refs:
        word2id[w] = tokenizer.encode([w])[1]
    return word2id


if __name__ == '__main__':

    from text_preprocessing import load_sents, generate_mask_sentence, generate_reference
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # print('NC:')
    # check_out_multitoken_words(tokenizer, data_flag='nc')
    # print('\nSCI:')
    # check_out_multitoken_words(tokenizer, data_flag='sci')

    nc_concepts, nc_sents = load_sents(data_flag='nc')
    sci_concepts, sci_sents = load_sents(data_flag='sci')

    sents = nc_sents + sci_sents
    print('num of all samples: ', len(sents))
    tokenized_text = tokenization(tokenizer, sents)

    local_vocab, word2id, unk_num, total_num = get_local_bert_vocab(tokenizer, tokenized_text)
    print('bert本地词表大小： ', len(local_vocab.keys()))
    print('oov词数量： ', unk_num)
    print('oov词占语料比例：{:.3%}'.format(unk_num / total_num))
    print('总词语数量： ', total_num)

    ref_word2id = get_refs_ids(tokenizer, nc_concepts, sci_concepts)
    vocab_df = save_local_vocab(local_vocab, word2id, ref_word2id)
    print(vocab_df.info())

