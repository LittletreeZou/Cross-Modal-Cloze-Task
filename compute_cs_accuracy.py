"""
Compute the retrieval accuracy of cross-modal mapping and representational similarity retrieval
"""

import os
import numpy as np
import pandas as pd
import argparse
import scipy.io as scio
from itertools import combinations
from utils.fmri_preprocessing import load_nc_fmri, load_sci_fmri, get_nc_fmri_words
from utils.text_preprocessing import load_text
from utils.data_util import cosine, word_kfold_split


# Z-score -- z-score each column
zscore = lambda v: (v-v.mean(0))/(v.std(0) + 1e-8)
# Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zscore(c1)*zscore(c2)).mean(0)
# Vector corr
zs = lambda x: (x-x.mean())/(x.std() + 1e-8)
corr = lambda x,y: (zs(x)*zs(y)).mean()


def compute_pair_corr(x1, x2, y1, y2, metric='corr'):
    if metric == 'corr':
        r11 = corr(x1, y1)
        r12 = corr(x1, y2)
        r21 = corr(x2, y1)
        r22 = corr(x2, y2)
    else:
        r11 = cosine(x1, y1)
        r12 = cosine(x1, y2)
        r21 = cosine(x2, y1)
        r22 = cosine(x2, y2)
    if (r11 + r22) >= (r12 + r21):
        return 1
    else:
        return 0


def pairwise_accuracy(wordvecs, predictions, val=True, remove=False, n_splits=10, metric='corr'):
    '''
    :param wordvecs: of shape (180, 768)
    :param predictions: of shape (180, 768)
    '''
    n_samples, dim = wordvecs.shape

    fold_index_list = word_kfold_split(n_samples, n_splits)

    all = 0
    correct = 0
    for k in range(n_splits):
        if val:
            test_index = fold_index_list[k]
        else:
            test_index = fold_index_list[(k + 1) % n_splits]
        for comb in combinations(test_index, 2):
            i = comb[0]
            j = comb[1]
            x1 = wordvecs[i]
            x2 = wordvecs[j]
            y1 = predictions[i]
            y2 = predictions[j]
            if remove:
                remove_index = [i, j]
                x1 = np.delete(x1, remove_index)
                x2 = np.delete(x2, remove_index)
                y1 = np.delete(y1, remove_index)
                y2 = np.delete(y2, remove_index)
            correct += compute_pair_corr(x1, x2, y1, y2, metric)
            all += 1
    print('Total number of pairs: ', all)
    print('Correct matching: ', correct)
    print('pairwise accuracy = {:.2%}'.format(correct/all))
    return correct/all


def topk_accuracy(wordvecs, predictions, remove=False, metric='corr'):
    '''
    60分类或者180分类
    '''
    n_samples = wordvecs.shape[0]
    corr_matrix = np.zeros((n_samples, n_samples))
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    top10_words_index = []
    for i in range(n_samples):
        pred = predictions[i]
        # compute the corr between pred and every wordvec
        for j in range(n_samples):
            if remove:
                remove_index = [i, j]
                pred1 = np.delete(pred, remove_index)
                wv1 = np.delete(wordvecs[j], remove_index)
                if metric == 'corr':
                    corr_matrix[i, j] = corr(pred1, wv1)
                else:
                    corr_matrix[i, j] = cosine(pred1, wv1)
            else:
                if metric == 'corr':
                    corr_matrix[i, j] = corr(pred, wordvecs[j])
                else:
                    corr_matrix[i, j] = cosine(pred, wordvecs[j])
        top10_index = np.argsort(-corr_matrix[i])[:10]
        top10_words_index.append(top10_index)
        if i in top10_index:
            top10_correct += 1
            if i in top10_index[:5]:
                top5_correct += 1
                if i == top10_index[0]:
                    top1_correct += 1

    top1_acc = top1_correct / n_samples
    top5_acc = top5_correct / n_samples
    top10_acc = top10_correct / n_samples
    print('top1 accuracy = {:.2%}'.format(top1_correct / n_samples))
    print('top5 accuracy = {:.2%}'.format(top5_correct / n_samples))
    print('top10 accuracy = {:.2%}'.format(top10_correct / n_samples))
    return top10_words_index, top1_acc, top5_acc, top10_acc


def pairwise_corr(fmri, metric='corr'):
    n = fmri.shape[0]
    result = np.ones((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            if metric == 'corr':
                c = corr(fmri[i], fmri[j])
            else:
                c = cosine(fmri[i], fmri[j])
            result[i, j] = c
            result[j, i] = c
    return result


# def RSA(fmri, split_folds, flag='test', metric='corr'):
#     '''
#     用相似度进行重新表征，去掉测试集的数据
#     '''
#     pairwise_sim_matrix = pairwise_corr(fmri, metric)
#     n = fmri.shape[0]
#     fold_size = n // len(split_folds)
#     indx = np.arange(n)
#
#     if flag == 'test':
#         repr = np.zeros((n, n-fold_size))
#         for k in range(len(split_folds)):
#             test_inx = split_folds[(k+1)%len(split_folds)]
#             train_inx = np.delete(indx, test_inx)
#             for i in test_inx:
#                 repr[i] = pairwise_sim_matrix[i, train_inx]
#     else:
#         repr = np.zeros((n, n-2*fold_size))
#         for k in range(len(split_folds)):
#             val_inx = split_folds[k]
#             test_inx = split_folds[(k + 1) % len(split_folds)]
#             train_inx = np.delete(indx, val_inx + test_inx)
#             for i in val_inx:
#                 repr[i] = pairwise_sim_matrix[i, train_inx]
#     return repr



def RSA_for_fold(fmri, wordvec, fold, split_folds, metric='corr'):
    '''
     The goal of rsa is to retrieve nearest neighbor word in the vocabulary for a val or test sample.

    以训练集的数据作为基，对数据进行用相似度进行重新表征
    '''
    val_inx = split_folds[fold]
    test_inx = split_folds[(fold + 1) % len(split_folds)]
    train_inx = np.delete(np.arange(fmri.shape[0]), val_inx + test_inx)

    val_repr = np.zeros((len(val_inx), len(train_inx)))
    for m, i in enumerate(val_inx):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(fmri[i], fmri[train_inx[j]])
            else:
                r[j] = cosine(fmri[i], fmri[train_inx[j]])
        val_repr[m] = r

    test_repr = np.zeros((len(test_inx), len(train_inx)))
    for m, i in enumerate(test_inx):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(fmri[i], fmri[train_inx[j]])
            else:
                r[j] = cosine(fmri[i], fmri[train_inx[j]])
        test_repr[m] = r

    # in the whole vocabulary
    wv_repr = np.zeros((wordvec.shape[0], len(train_inx)))
    for i in range(wordvec.shape[0]):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(wordvec[i], wordvec[train_inx[j]])
            else:
                r[j] = cosine(wordvec[i], wordvec[train_inx[j]])
        wv_repr[i] = r
    return val_repr, test_repr, wv_repr


def pairwise_accuracy_for_rsa_fold(fmri_repr, wv_repr, fold, split_folds, flag='test', metric='corr'):
    '''
    :param fmri_repr: (fold_size, train_size)
    :param wv_repr: (180, train_size)
    '''
    all = 0
    correct = 0
    if flag == 'test':
        test_inx = split_folds[(fold + 1) % len(split_folds)]
    else:
        test_inx = split_folds[fold]   # val_inx
    inx = [i for i in range(len(test_inx))]
    for comb in combinations(inx, 2):
        i = comb[0]
        j = comb[1]
        x1 = fmri_repr[i]
        x2 = fmri_repr[j]
        y1 = wv_repr[test_inx[i]]
        y2 = wv_repr[test_inx[j]]
        correct += compute_pair_corr(x1, x2, y1, y2, metric)
        all += 1
    acc = correct / all
    print('pairwise accuracy = {:.2%}'.format(acc))
    return acc


def topk_accuracy_for_rsa_fold(fmri_repr, wv_repr, fold, split_folds, flag='test', metric='corr'):
    '''
    60分类或者180分类
    '''
    n_samples = fmri_repr.shape[0]
    vocab_size = wv_repr.shape[0]
    corr_matrix = np.zeros((n_samples, vocab_size))
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    if flag == 'test':
        test_inx = split_folds[(fold + 1) % len(split_folds)]
    else:
        test_inx = split_folds[fold]  # val_inx
    for i in range(n_samples):
        r = fmri_repr[i]
        # compute the corr between r and every wordvec
        for j in range(vocab_size):
            if metric == 'corr':
                corr_matrix[i, j] = corr(r, wv_repr[j])
            else:
                corr_matrix[i, j] = cosine(r, wv_repr[j])
        top10_index = np.argsort(-corr_matrix[i])[:10]

        if test_inx[i] in top10_index:
            top10_correct += 1
            if test_inx[i] in top10_index[:5]:
                top5_correct += 1
                if test_inx[i] == top10_index[0]:
                    top1_correct += 1

    top1_acc = top1_correct / n_samples
    top5_acc = top5_correct / n_samples
    top10_acc = top10_correct / n_samples
    print('top1 accuracy = {:.2%}'.format(top1_acc))
    print('top5 accuracy = {:.2%}'.format(top5_acc))
    print('top10 accuracy = {:.2%}'.format(top10_acc))
    return top1_acc, top5_acc, top10_acc



def REG_RSA_for_fold(val_predictions, test_predictions, wordvecs, fold, split_folds, metric='corr'):
    '''
    注意，rsa在计算的时候，以ground-truth作为基
    '''
    val_inx = split_folds[fold]
    test_inx = split_folds[(fold + 1) % len(split_folds)]
    train_inx = np.delete(np.arange(wordvecs.shape[0]), val_inx + test_inx)

    val_repr = np.zeros((len(val_inx), len(train_inx)))
    for m, i in enumerate(val_inx):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(val_predictions[i], wordvecs[train_inx[j]])
            else:
                r[j] = cosine(val_predictions[i], wordvecs[train_inx[j]])
        val_repr[m] = r

    test_repr = np.zeros((len(test_inx), len(train_inx)))
    for m, i in enumerate(test_inx):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(test_predictions[i], wordvecs[train_inx[j]])
            else:
                r[j] = cosine(test_predictions[i], wordvecs[train_inx[j]])
        test_repr[m] = r

    # in the whole vocabulary
    wv_repr = np.zeros((wordvecs.shape[0], len(train_inx)))
    for i in range(wordvecs.shape[0]):
        r = np.zeros(len(train_inx))
        for j in range(len(train_inx)):
            if metric == 'corr':
                r[j] = corr(wordvecs[i], wordvecs[train_inx[j]])
            else:
                r[j] = cosine(wordvecs[i], wordvecs[train_inx[j]])
        wv_repr[i] = r
    return val_repr, test_repr, wv_repr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nc')
    parser.add_argument('--option', type=str, default='reg', help="reg, rsa, reg_rsa, reg_rsa_concat")
    args = parser.parse_args()

    DATA_FLAG = args.data
    print(DATA_FLAG)

    k = 5000       # for voxel selection, fixed
    wordvecs_types = ['bert_embed', 'bert_layeravg', 'glove']
    vs_wv = {'bert_embed': 'bert', 'bert_layeravg': 'bert', 'glove': 'glove'}


    if DATA_FLAG == 'nc':
        fmri_words, _, _ = get_nc_fmri_words(subject='M15')
        subjects = ["P01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M13", "M14", "M15", "M16",
                    "M17"]
    else:
        fmri_words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')
        subjects = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    word_folds = word_kfold_split(len(fmri_words), n_splits=10)


    for wordvecs_type in wordvecs_types:
        print(wordvecs_type)
        wv = vs_wv[wordvecs_type]    # for voxel selection
        if DATA_FLAG == 'nc':
            if wordvecs_type == 'glove':
                wordvecs = scio.loadmat('/home/sxzou/concept_decoding/data/wordvecs/nc/glove_180_words.mat')['glove_v1']
            else:
                wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/nc'
                wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_180_words.npy')
                wordvecs = np.load(wordvec_path)
        else:
            wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/sci'
            wordvec_path = os.path.join(wordvec_dir, wordvecs_type + '_60.npy')
            wordvecs = np.load(wordvec_path)

        if args.option == 'reg':
            accuracy = np.zeros((len(subjects), 8))  # 前面4列是valid, 后面4列是test
            for i, sub in enumerate(subjects):
                print(sub)
                if wv == 'bert':
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + sub \
                               + "/" + wordvecs_type + '/bert_5000'
                else:
                    # 64 server
                    # file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + sub \
                    #            + "/" + wordvecs_type + '/glove_5000'
                    # 58 server
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "_glove/" + sub \
                               + "/" + wordvecs_type + '/glove_5000'
                # valid
                predictions = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
                accuracy[i, 0] = pairwise_accuracy(wordvecs, predictions, val=True)
                accuracy[i, 1:4] = topk_accuracy(wordvecs, predictions)[1:]
                # test
                predictions = np.load(os.path.join(file_dir, 'predictions.npy'))
                accuracy[i, 4] = pairwise_accuracy(wordvecs, predictions, val=False)
                accuracy[i, 5:8] = topk_accuracy(wordvecs, predictions)[1:]
            accuracy = pd.DataFrame(accuracy)
            save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cs/' + DATA_FLAG + '/reg'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, wordvecs_type + '.csv')
            accuracy.to_csv(save_path, index=False)


        elif args.option == 'rsa':
            accuracy = np.zeros((len(subjects), 8))  # 前面4列是valid, 后面4列是test
            for i, sub in enumerate(subjects):
                print(sub)
                if DATA_FLAG == 'nc':
                    fmri = load_nc_fmri(paradigm='all', subject=sub, k=k, wv=wv)
                else:
                    fmri = load_sci_fmri(subject=sub, k=k, wv=wv)
                for fold in range(10):
                    val_r, test_r, wv_r = RSA_for_fold(fmri, wordvecs, fold, word_folds)
                    # valid
                    accuracy[i, 0] += pairwise_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    accuracy[i, 1:4] += topk_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    # test
                    accuracy[i, 4] += pairwise_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
                    accuracy[i, 5:8] += topk_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
            accuracy = pd.DataFrame(accuracy/10)
            save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cs/' + DATA_FLAG + '/rsa'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, wordvecs_type + '.csv')
            accuracy.to_csv(save_path, index=False)


        elif args.option == 'reg_rsa':
            accuracy = np.zeros((len(subjects), 8))  # 前面4列是valid, 后面4列是test
            for i, sub in enumerate(subjects):
                print(sub)
                if wv == 'bert':
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + sub \
                               + "/" + wordvecs_type + '/bert_5000'
                else:
                    # 64 server
                    # file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + sub \
                    #            + "/" + wordvecs_type + '/glove_5000'
                    # 58 server
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "_glove/" + sub \
                               + "/" + wordvecs_type + '/glove_5000'
                val_predictions = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
                test_predictions = np.load(os.path.join(file_dir, 'predictions.npy'))
                for fold in range(10):
                    val_r, test_r, wv_r = REG_RSA_for_fold(val_predictions, test_predictions, wordvecs, fold, word_folds)
                    # valid
                    accuracy[i, 0] += pairwise_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    accuracy[i, 1:4] += topk_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    # test
                    accuracy[i, 4] += pairwise_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
                    accuracy[i, 5:8] += topk_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
            accuracy = pd.DataFrame(accuracy/10)
            save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cs/' + DATA_FLAG + '/reg_rsa1'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, wordvecs_type + '.csv')
            accuracy.to_csv(save_path, index=False)


        elif args.option == 'reg_rsa_concat':
            accuracy = np.zeros((len(subjects), 8))  # 前面4列是valid, 后面4列是test
            for i, sub in enumerate(subjects):
                print(sub)
                if wv == 'bert':
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + sub \
                               + "/" + wordvecs_type + '/bert_5000'
                else:
                    # 64 server
                    # file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + sub \
                    #            + "/" + wordvecs_type + '/glove_5000'
                    # 58 server
                    file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "_glove/" + sub \
                               + "/" + wordvecs_type + '/glove_5000'
                val_predictions = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
                test_predictions = np.load(os.path.join(file_dir, 'predictions.npy'))
                for fold in range(10):
                    val_inx = word_folds[fold]
                    test_inx = word_folds[(fold + 1) % 10]
                    val_r, test_r, wv_r = REG_RSA_for_fold(val_predictions, test_predictions, wordvecs, fold, word_folds)
                    # concate rsa and reg represenations
                    val_r = np.concatenate([val_r, val_predictions[val_inx]], axis=1)
                    test_r = np.concatenate([test_r, test_predictions[test_inx]], axis=1)
                    wv_r = np.concatenate([wv_r, wordvecs], axis=1)
                    # valid
                    accuracy[i, 0] += pairwise_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    accuracy[i, 1:4] += topk_accuracy_for_rsa_fold(val_r, wv_r, fold, word_folds, flag='val')
                    # test
                    accuracy[i, 4] += pairwise_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
                    accuracy[i, 5:8] += topk_accuracy_for_rsa_fold(test_r, wv_r, fold, word_folds, flag='test')
            accuracy = pd.DataFrame(accuracy/10)
            save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cs/' + DATA_FLAG + '/reg_rsa_concat'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, wordvecs_type + '.csv')
            accuracy.to_csv(save_path, index=False)


        else:
            print('option is wrong')





        # # reg + rsa + csls
        # accuracy = np.zeros((len(subjects), 4))
        # wordvecs = RSA(wordvecs, word_folds, flag)
        # for i, sub in enumerate(subjects):
        #     print(sub)
        #     if flag == 'test':
        #         file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + DATA_FLAG + "/" + sub \
        #                    + "/" + wordvecs_type
        #         predictions = np.load(os.path.join(file_dir, 'predictions.npy'))
        #         predictions = RSA(predictions, word_folds, flag)
        #         accuracy[i, 0] = pairwise_accuracy(wordvecs, predictions, val=False, metric='corr')
        #     else:
        #         file_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/final/" + DATA_FLAG + "/" + sub \
        #                    + "/" + wordvecs_type + "/" + wv + '_' + str(k)
        #         predictions = np.load(os.path.join(file_dir, 'valid_predictions.npy'))
        #         predictions = RSA(predictions, word_folds, flag)
        #         accuracy[i, 0] = pairwise_accuracy(wordvecs, predictions, val=True, metric='corr')
        #     accuracy[i, 1:] = CSLS_topk_accuracy(wordvecs, predictions, metric='corr')[1:]
        # accuracy = pd.DataFrame(accuracy)
        # save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/cm_' + flag + '/' + DATA_FLAG + '/glove_' + str(
        #     k) + '/reg_rsa1_csls'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, wordvecs_type + '.csv')
        # accuracy.to_csv(save_path, index=False)
        #
        # # rsa + csls
        # accuracy = np.zeros((len(subjects), 4))
        # for i, sub in enumerate(subjects):
        #     print(sub)
        #     if DATA_FLAG == 'nc':
        #         fmri = load_nc_fmri(paradigm='all', subject=sub, k=k, wv=wv)
        #     else:
        #         fmri = load_sci_fmri(subject=sub, k=k, wv=wv)
        #     wordvecs_sim = RSA(wordvecs, word_folds, flag)
        #     fmri_sim = RSA(fmri, word_folds, flag)
        #     if flag == 'test':
        #         accuracy[i, 0] = pairwise_accuracy(wordvecs_sim, fmri_sim, val=False, remove=False, metric='corr')
        #     else:
        #         accuracy[i, 0] = pairwise_accuracy(wordvecs_sim, fmri_sim, val=True, remove=False, metric='corr')
        #     #accuracy[i, 1:] = topk_accuracy(wordvecs_sim, fmri_sim, remove=False, metric='corr')[1:]
        #     accuracy[i, 1:] = CSLS_topk_accuracy(wordvecs_sim, fmri_sim, metric='corr', remove=False)[1:]
        # accuracy = pd.DataFrame(accuracy)
        # save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/cm_' + flag + '/' + DATA_FLAG + '/glove_' + str(
        #     k) + '/rsa1_csls'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, wordvecs_type + '.csv')
        # accuracy.to_csv(save_path, index=False)

