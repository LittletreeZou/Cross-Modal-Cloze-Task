import os
import numpy as np
import pandas as pd
import argparse
import scipy.io as scio
import torch
from joblib import Parallel, delayed   # for parallel
from transformers import BertTokenizer
from utils.small_bert import get_restricted_vocab_for_bert, SmallBertForMaskedLM
from utils.text_preprocessing import load_text, load_sents, generate_mask_sentence, generate_reference
from utils.fmri_preprocessing import get_nc_fmri_words, load_nc_fmri, load_sci_fmri
from utils.data_util import word_kfold_split, sent_kfold_split, sid_to_wid, cosine
from compute_cs_accuracy import corr, REG_RSA_for_fold, RSA_for_fold
import warnings
warnings.filterwarnings(action='ignore')


# for retrieval
def get_topk_preds(words, bertvecs, wordvecs, fmri_embed, topk=5, metric='corr'):
    '''
    对每个fmri对齐后的向量，获取与它相关系数top k的词语，及其词向量
    bertvecs: (n, 768)
    wordvecs: (n, dim)
    fmri_embed: (m, dim)
    '''
    n_samples = fmri_embed.shape[0]
    topk_words = []
    topk_wvs = np.zeros((n_samples, bertvecs.shape[1]))

    for i in range(n_samples):
        fmri = fmri_embed[i]
        sim = np.zeros(len(words))
        # compute the corr between pred and every wordvec
        for j in range(len(words)):
            wv = wordvecs[j]
            if metric == 'corr':
                sim[j] = corr(fmri, wv)
            else:
                sim[j] = cosine(fmri, wv)
        topk_index = np.argsort(-sim)[:topk]  # sord in descending order
        topk_words.append([words[k] for k in topk_index])
        # average the topk word embedding 
        topk_wvs[i] = bertvecs[topk_index].mean(axis=0)
    return topk_words, topk_wvs




def get_mask_predict_for_single_sent(model, tokenizer, token_ids, bertid2localid,
                                     mask_sent, ref, feature, alpha=0.8, topk=10):
    '''
    feature: contains information about answer word
    alpha: controls the information that we fuse in
    '''
    input_ids = tokenizer(mask_sent, return_tensors='pt')['input_ids']
    masked_index = torch.nonzero(input_ids[0] == tokenizer.mask_token_id, as_tuple=False).item()
    input_ids.apply_(lambda x: bertid2localid[x])
    input_ids = input_ids.cuda()
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(input_ids)  #(batch_size, seq_len, 768)
        input_embeds[0, masked_index] = (1-alpha)*input_embeds[0, masked_index] + alpha*torch.tensor(feature, dtype=torch.float32).cuda()
        logits = model(inputs_embeds=input_embeds).logits
    probs = logits[0, masked_index].softmax(dim=0)  # our vocab_size
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


def get_mask_predict(model, tokenizer, token_ids, bertid2rlocalid,
                     mask_sents, refs, features, alpha):
    '''
    获取模型对mask预测的top1, top5, top 10准确率及目标词的预测概率
    '''
    n_samples = len(refs)

    preds = []
    top1_acc = np.zeros(n_samples)
    top5_acc = np.zeros(n_samples)
    top10_acc = np.zeros(n_samples)
    target_word_prob = np.zeros(n_samples)

    for i, mask_sent in enumerate(mask_sents):
        ref = refs[i]
        topk_words, answer_word, answer_prob = get_mask_predict_for_single_sent(model, tokenizer, token_ids, bertid2rlocalid,
                                                                                mask_sent, ref, features[i], alpha, topk=10)
        preds.append(topk_words)
        target_word_prob[i] = answer_prob

        if answer_word in topk_words:
            top10_acc[i] = 1
            if answer_word in topk_words[:5]:
                top5_acc[i] = 1
                if answer_word == topk_words[0]:
                    top1_acc[i] = 1

    all_accs = [top1_acc.mean(), top5_acc.mean(), top10_acc.mean(), target_word_prob.mean()]
    return preds, all_accs



def kfold_cv(tokenizer, sent_folds, mask_sents, references, features):
    '''
    search for the best alpha and k using "top5_acc" as metric on valid data

    :param mask_sents: n_samples
    :param references: n_samples, list of list
    :param features: feature vector for each sentence
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

    # hyperparameter
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    valid_top5_accs = np.zeros((1, len(alpha_list)))
    for i, alpha in enumerate(alpha_list):
        for fold in range(len(sent_folds)):
            #print('\nfold ' + str(fold))
            valid_index = sent_folds[fold]
            valid_sents = [mask_sents[i] for i in valid_index]
            valid_refs = [references[i] for i in valid_index]
            valid_features = [features[i] for i in valid_index]
            _, all_accs = get_mask_predict(model, tokenizer, token_ids, bertid2localid,
                                           valid_sents, valid_refs, valid_features, alpha)
            valid_top5_accs[0, i] += all_accs[1]
    valid_top5_accs /= 10
    #best_alpha = alpha_list[valid_top5_accs.argmax()]
    return valid_top5_accs


def single_subject_tuning(wordvecs_type, subject, k=None, data_flag='nc', direct_fusion=False):
    '''
    return scores top-5 acc for each combinations of (k, alpha)
    '''
    # load fmri predictions
    if wordvecs_type == 'glove':
        file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + subject \
                   + "/" + wordvecs_type + '/glove_5000'
    else:
        file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + subject \
                   + "/" + wordvecs_type + '/bert_5000'
    val_wordfmri = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
    test_wordfmri = np.load(os.path.join(file_dir, 'predictions.npy'))

    if direct_fusion:
        sentfmri = [val_wordfmri[sid2wid[i]] for i in range(len(concepts))]
    else:
        # load wordvecs
        if data_flag == 'nc':
            bertvecs = np.load('/home/sxzou/concept_decoding/data/wordvecs/nc/bert_embed_180_words.npy')
            if wordvecs_type == 'glove':
                wordvecs = scio.loadmat('/home/sxzou/concept_decoding/data/wordvecs/nc/glove_180_words.mat')['glove_v1']
            else:
                wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/nc'
                wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_180_words.npy')
                wordvecs = np.load(wordvec_path)
        else:
            bertvecs = np.load('/home/sxzou/concept_decoding/data/wordvecs/sci/bert_embed_60.npy')
            wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/sci'
            wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_60.npy')
            wordvecs = np.load(wordvec_path)
        # get fmri features for 10 fold
        topk_wvs = [0 for _ in range(wordvecs.shape[0])]
        for fold in range(10):
            val_r, _, wv_r = REG_RSA_for_fold(val_wordfmri, test_wordfmri, wordvecs, fold, word_folds)
            _, fold_topk_wvs = get_topk_preds(fmri_words, bertvecs, wv_r, val_r, topk=k)
            val_inx = word_folds[fold]
            for i, inx in enumerate(val_inx):
                topk_wvs[inx] = fold_topk_wvs[i]
        sentfmri = [topk_wvs[sid2wid[i]] for i in range(len(concepts))]
    valid_top5_accs = kfold_cv(tokenizer, sent_folds, mask_sents, references, sentfmri)
    return valid_top5_accs


def parallel_tuning(wordvecs_type, subjects, k=None, data_flag='nc', direct_fusion=False):
    all_accs = Parallel(n_jobs=10)(
        delayed(single_subject_tuning)(wordvecs_type, sub, k, data_flag, direct_fusion)
        for sub in subjects)
    return np.concatenate(all_accs, axis=0)




def kfold_evaluate(tokenizer, sent_folds, mask_sents, references, features, alpha=0.7):
    '''
    :param mask_sents: n_samples
    :param references: n_samples(按顺序排列）
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
        test_features = [features[i] for i in test_index]
        preds, all_accs = get_mask_predict(model, tokenizer, token_ids, bertid2localid,
                                            test_sents, test_refs, test_features, alpha)

        top10_preds.extend(preds)
        accuracy[:, k] = all_accs

    accuracy = pd.DataFrame(accuracy)
    # save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/cloze/' + data_flag
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, 'baseline.csv')
    # accuracy.to_csv(save_path, index=False)
    return top10_preds, accuracy


def single_subject_evaluate(wordvecs_type, subject, k=None, alpha=None, data_flag='nc', direct_fusion=False):

    # load fmri predictions
    if wordvecs_type == 'glove':
        file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + subject \
                   + "/" + wordvecs_type + '/glove_5000'
    else:
        file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + subject \
                   + "/" + wordvecs_type + '/bert_5000'
    val_wordfmri = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
    test_wordfmri = np.load(os.path.join(file_dir, 'predictions.npy'))

    if direct_fusion:
        sentfmri = [test_wordfmri[sid2wid[i]] for i in range(len(concepts))]
    else:
        # load wordvecs
        if data_flag == 'nc':
            bertvecs = np.load('/home/sxzou/concept_decoding/data/wordvecs/nc/bert_embed_180_words.npy')
            if wordvecs_type == 'glove':
                wordvecs = scio.loadmat('/home/sxzou/concept_decoding/data/wordvecs/nc/glove_180_words.mat')['glove_v1']
            else:
                wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/nc'
                wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_180_words.npy')
                wordvecs = np.load(wordvec_path)
        else:
            bertvecs = np.load('/home/sxzou/concept_decoding/data/wordvecs/sci/bert_embed_60.npy')
            wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/sci'
            wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_60.npy')
            wordvecs = np.load(wordvec_path)
        # get fmri features for 10 fold
        topk_wvs = [0 for _ in range(wordvecs.shape[0])]
        for fold in range(10):
            # REG_RSA
            if args.retri == 'reg_rsa':
                _, test_r, wv_r = REG_RSA_for_fold(val_wordfmri, test_wordfmri, wordvecs, fold, word_folds)
                _, fold_topk_wvs = get_topk_preds(fmri_words, bertvecs, wv_r, test_r, topk=int(k))

            elif args.retri == 'rsa':
                if DATA_FLAG == 'nc':
                    fmri = load_nc_fmri(paradigm='all', subject=subject, k=5000, wv='bert')
                else:
                    fmri = load_sci_fmri(subject=subject, k=5000, wv='bert')
                _, test_r, wv_r = RSA_for_fold(fmri, wordvecs, fold, word_folds)
                _, fold_topk_wvs = get_topk_preds(fmri_words, bertvecs, wv_r, test_r, topk=int(k))

            elif args.retri == 'reg':
                _, fold_topk_wvs = get_topk_preds(fmri_words, bertvecs, wordvecs, test_wordfmri, topk=int(k))

            else:
                print('wrong arg.retri!')

            test_inx = word_folds[(fold+1)%10]
            for i, inx in enumerate(test_inx):
                topk_wvs[inx] = fold_topk_wvs[i]
        sentfmri = [topk_wvs[sid2wid[i]] for i in range(len(concepts))]

    if args.random:
        np.random.seed(args.seed)
        np.random.shuffle(sentfmri)
    _, accs = kfold_evaluate(tokenizer, sent_folds, mask_sents, references, sentfmri, alpha)
    return accs


def parallel_evaluate(wordvecs_type, subjects, data_flag='nc', direct_fusion=False):

    # load best hyperparameters
    para_path = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/tuning/' + data_flag + '_params.csv'
    para = pd.read_csv(para_path, header=None).values  # (5, 9)
    if wordvecs_type == 'bert_embed':
        ks = [None]*len(subjects)
        #alphas = para[0]
        alphas = [0.7]*len(subjects)
    elif wordvecs_type == 'bert_layeravg':
        # ks = para[1]
        # alphas = para[2]
        ks = [5]*len(subjects)
        alphas = [0.7]*len(subjects)
    else:
        # ks = para[3]
        # alphas = para[4]
        ks = [5] * len(subjects)
        alphas = [0.7] * len(subjects)
    all_accs = Parallel(n_jobs=10)(delayed(single_subject_evaluate)(wordvecs_type, sub, ks[i], alphas[i], data_flag, direct_fusion)
                                   for i, sub in enumerate(subjects))
    return np.concatenate(all_accs)


def parallel_analysis(data_flag='nc'):

    # avg_accs_for_alpha = np.zeros((11, 4))
    # ks = [5]*len(subjects)
    # for i, alpha in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    #     alphas = [alpha]*len(subjects)
    #     all_accs = Parallel(n_jobs=10)(delayed(single_subject_evaluate)(wordvecs_type, sub, ks[i], alphas[i], data_flag, direct_fusion=False)
    #                                    for i, sub in enumerate(subjects))
    #     accs = np.mean(all_accs, axis=0)  # average across subjects
    #     accs = accs.mean(axis=1)   # average across folds
    #     avg_accs_for_alpha[i] = accs
    # final_accs = avg_accs_for_alpha

    avg_accs_for_k = np.zeros((3, 4))
    alphas = [0.7] * len(subjects)
    #for i, k in enumerate([1, 3, 5, 10, 12]):
    for i, k in enumerate([3, 8, 12]):
        ks = [k] * len(subjects)
        all_accs = Parallel(n_jobs=10)(
            delayed(single_subject_evaluate)(wordvecs_type, sub, ks[i], alphas[i], data_flag, direct_fusion=False)
            for i, sub in enumerate(subjects))
        accs = np.mean(all_accs, axis=0)  # average across subjects
        accs = accs.mean(axis=1)  # average across folds
        avg_accs_for_k[i] = accs
    final_accs = avg_accs_for_k

    return final_accs



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='sci')
    parser.add_argument('--wordvec', type=str, default='bert_embed', help="glove, bert_layeravg, bert_embed")
    parser.add_argument('--tuning', action='store_true', help='to search for hyperparameters')
    parser.add_argument('--random', action='store_true', help='shuffle fmri data to make them mismatch')
    parser.add_argument('--seed', type=int, default=567)
    parser.add_argument('--retri', type=str, default='reg_rsa')
    args = parser.parse_args()

    DATA_FLAG = args.data
    print(DATA_FLAG)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    wordvecs_type = args.wordvec
    vs_wv = {'bert_embed': 'bert', 'bert_layeravg': 'bert', 'glove': 'glove'}

    # load data
    if DATA_FLAG == 'nc':
        fmri_words, _, _ = get_nc_fmri_words(subject='M15')
        subjects = ["P01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M13", "M14", "M15", "M16",
                    "M17"]
    else:
        fmri_words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
        subjects = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    concepts, sents = load_sents(data_flag=DATA_FLAG)
    mask_words, mask_sents = generate_mask_sentence(concepts, sents, mask_token='[MASK]')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    references = generate_reference(concepts, mask_words, tokenizer, data_flag=DATA_FLAG)  # 包含了同义词


    # split data into 10 folds
    sid2wid = sid_to_wid(concepts, fmri_words)
    word_folds = word_kfold_split(len(fmri_words), n_splits=10)
    sent_folds = sent_kfold_split(word_folds, sid2wid)

    if args.tuning:
        # choosing the best alpha and k
        if wordvecs_type == 'bert_embed':
            # direct fusion
            accs = parallel_tuning(wordvecs_type, subjects, k=None, data_flag=DATA_FLAG, direct_fusion=True)
            accs = pd.DataFrame(accs)
            save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/tuning/' + DATA_FLAG
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, wordvecs_type + '.csv')
            accs.to_csv(save_path, index=False)
        else:
            k_list = [1, 5, 10]
            #k_list = [3, 12]
            for k in k_list:
                accs= parallel_tuning(wordvecs_type, subjects, k, data_flag=DATA_FLAG, direct_fusion=False)
                accs = pd.DataFrame(accs)
                save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/tuning/' + DATA_FLAG
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, wordvecs_type + '_' + str(k) + '.csv')
                accs.to_csv(save_path, index=False)

    else:
        # load hyperparamters
        # nc_alphas = {'bert_layeravg': [0.8, 0.8, 0.8, 0.9, 0.8, 0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.7, 0.8, 0.8, 0.7],
        #              'glove': [0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.7, 0.8, 0.7, 0.7],
        #              'bert_embed': [0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7]}
        # sci_alphas = {'bert_layeravg': [0.9, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7],
        #              'glove': [0.8, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8],
        #              'bert_embed': [0.8, 0.8, 0.8, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7]}

        if wordvecs_type == 'bert_embed':
            # direct fusion
            accuracy = parallel_evaluate(wordvecs_type, subjects, data_flag=DATA_FLAG, direct_fusion=True)
        else:
            accuracy = parallel_evaluate(wordvecs_type, subjects, data_flag=DATA_FLAG, direct_fusion=False)
        accuracy = pd.DataFrame(accuracy)
        save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/' + DATA_FLAG + '/fusion'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.random:
            if wordvecs_type == 'bert_embed':
                save_path = os.path.join(save_dir, 'random_direct_' + wordvecs_type + '_seed' + str(args.seed) + '_alpha0.7.csv')
            else:
                save_path = os.path.join(save_dir, 'random_retri_' + wordvecs_type + '_seed' + str(args.seed) + '_k5_alpha0.7.csv')
        else:
            if wordvecs_type == 'bert_embed':
                save_path = os.path.join(save_dir, 'direct_' + wordvecs_type + '_alpha0.7.csv')
            else:
                save_path = os.path.join(save_dir, args.retri + '_retri_' + wordvecs_type + '_k5_alpha0.7.csv')
        accuracy.to_csv(save_path, index=False)


    # analysis
    acc = parallel_analysis(data_flag=DATA_FLAG)
    accuracy = pd.DataFrame(acc)
    #save_path = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/analysis/' + DATA_FLAG + '_alphas.csv'
    save_path = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze/analysis/' + DATA_FLAG + '_ks_new.csv'
    accuracy.to_csv(save_path, index=False)
