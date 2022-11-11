import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from utils.small_bert import get_restricted_vocab_for_bert, SmallBertForMaskedLM
from utils.text_preprocessing import load_text, load_sents, generate_mask_sentence, generate_reference
from utils.fmri_preprocessing import get_nc_fmri_words
from utils.data_util import word_kfold_split, sent_kfold_split, sid_to_wid
import warnings
warnings.filterwarnings(action='ignore')



def get_mask_predict_for_single_sent(model, tokenizer, token_ids, bertid2localid, mask_sent, ref, topk=10):

    input_ids = tokenizer(mask_sent, return_tensors='pt')['input_ids']
    masked_index = torch.nonzero(input_ids[0] == tokenizer.mask_token_id, as_tuple=False).item()
    input_ids.apply_(lambda x: bertid2localid[x])
    input_ids = input_ids.cuda()
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(input_ids)  # (batch_size, seq_len, 768)
        logits = model(inputs_embeds=input_embeds).logits
    probs = logits[0, masked_index].softmax(dim=0)              # our vocab_size
    values, predictions = probs.topk(topk)

    # get answer probability
    probs = probs.cpu().numpy()
    answer_prob = 0
    answer_id = -1
    for answer in ref:
        answer_ids = bertid2localid[tokenizer.encode([answer])[1]]
        if probs[answer_ids] > answer_prob:
            answer_prob = probs[answer_ids]
            answer_id = answer_ids
    answer_word = tokenizer.decode([token_ids[answer_id]])

    predictions = predictions.cpu()
    predictions.apply_(lambda x: token_ids[x])
    topk_words = [tokenizer.decode([predictions[i]]) for i in range(len(predictions))]
    return topk_words, answer_word, answer_prob


def get_mask_predict(model, tokenizer, token_ids, bertid2localid, mask_sents, refs):
    '''
    predict the mask tokens and return top1, top5, top10 accuracy and target word probability
    '''
    n_samples = len(refs)

    preds = []
    top1_acc = np.zeros(n_samples)
    top5_acc = np.zeros(n_samples)
    top10_acc = np.zeros(n_samples)
    target_word_prob = np.zeros(n_samples)

    for i, mask_sent in enumerate(mask_sents):
        ref = refs[i]
        topk_words, answer_word, answer_prob = get_mask_predict_for_single_sent(model, tokenizer, token_ids, bertid2localid,
                                                                                mask_sent, ref, topk=10)
        preds.append(topk_words)
        target_word_prob[i] = answer_prob
        if answer_word in topk_words:
            top10_acc[i] = 1
            if answer_word in topk_words[:5]:
                top5_acc[i] = 1
                if answer_word == topk_words[0]:
                    top1_acc[i] = 1

    all_accs = [top1_acc.mean(), top5_acc.mean(), top10_acc.mean(),
                target_word_prob.mean()]
    return preds, all_accs


def kfold_evaluate(tokenizer, mask_sents, references, sent_folds, data_flag='nc'):
    '''
    :param mask_sents: n_samples
    :param references: n_samples(sort in order）, list of list
    '''
    # tokenizer
    special_ids = [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id,
                   tokenizer.pad_token_id, tokenizer.unk_token_id]
    token_ids, bertid2localid = get_restricted_vocab_for_bert(special_ids)

    # model
    model = SmallBertForMaskedLM.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(new_num_tokens=len(token_ids))
    model.cuda()
    model.eval()

    top10_preds = []
    accuracy = np.zeros((4, len(sent_folds)))

    for k in range(len(sent_folds)):
        print('\nfold ' + str(k))
        test_index = sent_folds[(k + 1) % len(sent_folds)]
        test_sents = [mask_sents[i] for i in test_index]
        test_refs = [references[i] for i in test_index]
        preds, all_accs = get_mask_predict(model, tokenizer, token_ids, bertid2localid,
                                           test_sents, test_refs)
        top10_preds.extend(preds)
        accuracy[:, k] = all_accs

    accuracy = pd.DataFrame(accuracy)
    save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/cloze/' + data_flag
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'baseline.csv')
    accuracy.to_csv(save_path, index=False)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    DATA_FLAG = 'sci'

    # load data
    if DATA_FLAG == 'nc':
        fmri_words, _, _ = get_nc_fmri_words(subject='M15')  #180
    else:
        fmri_words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')  #60
    concepts, sents = load_sents(data_flag=DATA_FLAG)
    mask_words, mask_sents = generate_mask_sentence(concepts, sents, mask_token='[MASK]')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    references = generate_reference(concepts, mask_words, tokenizer, data_flag=DATA_FLAG)

    # split data into 10 folds
    sid2wid = sid_to_wid(concepts, fmri_words)
    word_folds = word_kfold_split(len(fmri_words), n_splits=10)
    sent_folds = sent_kfold_split(word_folds, sid2wid)

    kfold_evaluate(tokenizer, mask_sents, references, sent_folds, data_flag=DATA_FLAG)
