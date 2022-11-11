import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


cosine = lambda x,y: np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y) + 1e-8)


def get_answer_ids(mask_word, sent, tokenizer):
    answer_ids = None
    input_ids = tokenizer.encode(sent)
    word_ids = tokenizer.encode(mask_word)[1:-1]
    n = len(word_ids)
    for i in range(len(input_ids)):
        if input_ids[i: i+n] == word_ids:
            answer_ids = word_ids
    if answer_ids is None:
        word_ids = tokenizer.encode(mask_word)[1:-1]
        n = len(word_ids)
        for i in range(len(input_ids)):
            if input_ids[i: i+n] == word_ids:
                answer_ids = word_ids
    if answer_ids is None:
        print(mask_word, sent)
        print(word_ids, input_ids)
    return answer_ids


def encode(mask_sents, sents, tokenizer, max_len, bertid2localid):
    '''
    tokenize data
    :param mask_sents: model input
    :param sents: labels for model
    '''
    pad_token_id = tokenizer.pad_token_id

    input_ids = []
    input_attention_mask = []
    labels = []
    sample_ind = []

    for i, sent in enumerate(mask_sents):
        # for input text
        encodings = tokenizer(sent, max_length=max_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
        input = encodings['input_ids']
        input.apply_(lambda x: bertid2localid[x])  # Map the bart vocab id to our local vocab id
        input_ids.append(input)
        input_attention_mask.append(encodings['attention_mask'])

        # for target text
        encodings = tokenizer(sents[i], max_length=max_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
        label = encodings['input_ids']
        label.apply_(lambda x: bertid2localid[x])  # Map the bart vocab id to our local vocab id
        label[label[:, :] == pad_token_id] = -100
        labels.append(label)
        sample_ind.append(i)

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    input_attention_mask = torch.cat(input_attention_mask, dim=0)
    labels = torch.cat(labels, dim=0)
    sample_ind = torch.tensor(sample_ind, dtype=torch.int32)

    dataset = TensorDataset(input_ids,
                            input_attention_mask,
                            labels,
                            sample_ind)
    return dataset


def get_dataloader(dataset, batch_size, random_sample=True):
    if random_sample:
        dataloader = DataLoader(dataset,
                                sampler = RandomSampler(dataset),
                                batch_size = batch_size)
    else:
        dataloader = DataLoader(dataset,
                                sampler = SequentialSampler(dataset),
                                batch_size = batch_size)
    return dataloader


def get_mask_index(input_ids, mask_token_id):
    '''
    :param input_ids: tensor of shape (batch_size, seq_len)
    :param tokenizer: bart tokenizer
    :return: mask_index, tensor of shape (batch_size, 2)
    '''
    masked_index_0 = list((input_ids == mask_token_id).nonzero()[:, 0].cpu().numpy())
    masked_index_1 = list((input_ids == mask_token_id).nonzero()[:, 1].cpu().numpy())
    return masked_index_0, masked_index_1



def sid_to_wid(concepts, fmri_words):
    '''
    Map sentence index to its concept word index
    :param concepts: of the same length as the numbers of sentences
    :param fmri_words: words corresponding to fmri index
    '''
    sid2wid = []
    for word in concepts:
        sid2wid.append(fmri_words.index(word))
    return sid2wid


def wid_to_sid(sid2wid):
    '''
    Get all the sentences for each concept word
    '''
    wid2sid = {}
    for sid, wid in enumerate(sid2wid):
        try:
            wid2sid[wid].append(sid)
        except KeyError:
            wid2sid[wid] = [sid]
    return wid2sid


def word_kfold_split(num_samples, n_splits=10, seed=123):
    '''
    split words into n_splits folds and return the indexes in each fola
    '''
    index = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(index)
    n = int(num_samples / n_splits)
    split_index_list = []
    for i in range(n_splits-1):
        fold_index = list(index[n*i: n*(i+1)])
        split_index_list.append(fold_index)
    fold_index = list(index[n*(n_splits-1):])
    split_index_list.append(fold_index)
    return split_index_list


def sent_kfold_split(split_word_index_list, sid2wid):
    '''
    split sentences into given fold according to the word_split
    '''
    wid2sid = wid_to_sid(sid2wid)
    split_sent_index_list = []
    for i, fold_word_index in enumerate(split_word_index_list):
        # find the corresponding sentences
        fold_sent_index = []
        for wid in fold_word_index:
            fold_sent_index.extend(wid2sid[wid])
        assert len(fold_sent_index) == len(fold_word_index)*6, 'sentence number is wrong!'
        split_sent_index_list.append(fold_sent_index)
    return split_sent_index_list