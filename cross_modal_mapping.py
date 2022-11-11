'''
统一10折交叉验证 2021.11.4
'''

import argparse
import os
import time
import numpy as np
import pandas as pd
import scipy.io as scio
from utils.text_preprocessing import format_time
from utils.fmri_preprocessing import load_nc_fmri, load_sci_fmri
from utils.ridge_tools import ridge, ridge_by_lambda
from utils.data_util import word_kfold_split

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from joblib import Parallel, delayed   # for parallel training


# Z-score -- z-score each column
zscore = lambda v: (v-v.mean(0))/(v.std(0) + 1e-8)
# Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zscore(c1)*zscore(c2)).mean(0)


def single_decoder(reg, X_train, y_train, X_valid, save_weights=False):
    '''
    :param reg: sklearn regressor
    :param X_train: (n, 5000)
    :param Y_train: (n,)
    '''
    n_voxels = X_train.shape[1]

    reg.fit(X_train, y_train)
    y_valid = reg.predict(X_valid)

    if save_weights:
        weights = np.zeros(n_voxels+1)
        weights[:n_voxels] = reg.coef_
        weights[-1] = reg.intercept_
        return y_valid, weights
    else:
        return y_valid


def kfold_cv_sklearn_regressor(args, fmri, wordvecs, n_splits, save_dir):
    '''
    对lasso /  elastic / svr / mlp / sgd 进行18折交叉验证

    1. search for best hyper parameters
    2. fit model with train+valid
    3. predict on test
    '''
    n_samples, n_voxels = fmri.shape
    n_targets = wordvecs.shape[1]

    predictions = np.zeros((n_samples, n_targets))
    fold_index_list = word_kfold_split(n_samples, n_splits)

    if args.regressor == 'lasso':
        for k in range(n_splits):
            print('\nFold {}:'.format(k))
            val_index = fold_index_list[k]
            test_index = fold_index_list[(k+1) % n_splits]
            train_index = list(set(np.arange(n_samples)) - set(val_index + test_index))
            train_x = fmri[train_index]
            train_y = wordvecs[train_index]
            val_x = fmri[val_index]
            val_y = wordvecs[val_index]

            # hyper parameter searching
            alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
            corrs = np.zeros((len(alphas), n_targets))
            for j, alpha in enumerate(alphas):
                reg = linear_model.LassoLars(alpha=alpha)  # LARS 最小角回归
                # preds is a list of length n_targets
                preds = Parallel(n_jobs=args.njobs)(
                           delayed(single_decoder)(reg, train_x, train_y[:, i], val_x) for i in range(n_targets))
                preds = np.array(preds).T
                #print(preds)
                assert preds.shape == val_y.shape, 'prediction shape error'
                corrs[j] = mcorr(preds, val_y)
            best_alpha_index = np.argmax(corrs, axis=0)
            print(pd.Series(best_alpha_index).value_counts())
            # save the correlations
            saved_corr = corrs.max(axis=0)
            save_path = os.path.join(save_dir, 'valid_correlations_fold_' + str(k))
            np.save(save_path, saved_corr)
            # refit model on train+valid and predict on test using the best alpha
            train_x = np.concatenate([train_x, val_x])
            train_y = np.concatenate([train_y, val_y])
            test_x = fmri[test_index]
            regs = []
            for i in range(n_targets):
                idx = best_alpha_index[i]
                best_alpha = alphas[idx]
                reg = linear_model.LassoLars(alpha=best_alpha)
                regs.append(reg)
            # list of tuples (y_valid, weights)
            test_preds = Parallel(n_jobs=args.njobs)(
                                  delayed(single_decoder)(regs[i], train_x, train_y[:, i], test_x, save_weights=True)
                                  for i in range(n_targets))
            weights = np.array([test_preds[i][1] for i in range(n_targets)])
            assert weights.shape == (n_targets, n_voxels+1), 'weights shape error'
            save_path = os.path.join(save_dir, 'reg_weights_fold_' + str(k))
            np.save(save_path, weights)
            print('weights save to ', save_path)
            preds = np.array([test_preds[i][0] for i in range(n_targets)]).T
            assert preds.shape == (len(test_index), n_targets), 'test preds shape error'
            predictions[test_index] = preds
        save_path = os.path.join(save_dir, 'predictions')
        np.save(save_path, predictions)
        print('predictions save to ', save_path)
        print('done')

    elif args.regressor == 'elastic':
        for k in range(n_splits):
            print('\nFold {}:'.format(k))
            val_index = fold_index_list[k]
            test_index = fold_index_list[(k+1) % n_splits]
            train_index = list(set(np.arange(n_samples)) - set(val_index + test_index))
            train_x = fmri[train_index]
            train_y = wordvecs[train_index]
            val_x = fmri[val_index]
            val_y = wordvecs[val_index]

            # hyper parameter searching
            alphas = [0.01, 0.1, 1, 10, 100]
            l1_ratios = [0.1, 0.5]
            params = [(i, j) for i in alphas for j in l1_ratios]
            print(params)
            corrs = np.zeros((len(params), n_targets))
            for j, param in enumerate(params):
                reg = linear_model.ElasticNet(alpha=param[0], l1_ratio=param[1], max_iter=10000,
                                              selection='random', random_state=1)
                preds = Parallel(n_jobs=args.njobs)(
                           delayed(single_decoder)(reg, train_x, train_y[:, i], val_x) for i in range(n_targets))
                preds = np.array(preds).T
                corrs[j] = mcorr(preds, val_y)
            best_param_index = np.argmax(corrs, axis=0)
            print(pd.Series(best_param_index).value_counts())
            # save the correlations
            saved_corr = corrs.max(axis=0)
            save_path = os.path.join(save_dir, 'valid_correlations_fold_' + str(k))
            np.save(save_path, saved_corr)
            # refit model on train+valid and predict on test using the best alpha
            train_x = np.concatenate([train_x, val_x])
            train_y = np.concatenate([train_y, val_y])
            test_x = fmri[test_index]
            regs = []
            for i in range(n_targets):
                idx = best_param_index[i]
                best_param = params[idx]
                reg = linear_model.ElasticNet(alpha=best_param[0], l1_ratio=best_param[1], max_iter=10000,
                                              selection='random', random_state=1)
                regs.append(reg)
            # list of tuples (y_valid, weights)
            test_preds = Parallel(n_jobs=args.njobs)(
                                  delayed(single_decoder)(regs[i], train_x, train_y[:, i], test_x, save_weights=True)
                                  for i in range(n_targets))
            weights = np.array([test_preds[i][1] for i in range(n_targets)])
            assert weights.shape == (n_targets, n_voxels+1), 'weights shape error'
            save_path = os.path.join(save_dir, 'reg_weights_fold_' + str(k))
            np.save(save_path, weights)
            print('weights save to ', save_path)
            preds = np.array([test_preds[i][0] for i in range(n_targets)]).T
            assert preds.shape == (len(test_index), n_targets), 'test preds shape error'
            predictions[test_index] = preds
        save_path = os.path.join(save_dir, 'predictions')
        np.save(save_path, predictions)
        print('predictions save to ', save_path)
        print('done')
    else:
        raise Exception('model not existed')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', action='store_true', help="perform encoding instead of decoding")
    parser.add_argument('--subject', type=str, default='M15', help="fmri subject")
    parser.add_argument('--regressor', type=str, default='ridge', help="regressor to be used")
    parser.add_argument('--wordvecs', type=str, default='glove', help='type of word vectors used')
    parser.add_argument('--njobs', type=int, default=5, help="number of parallel jobs")
    parser.add_argument('--save_dir', type=str,default=None, help='path to save weights and predictions')
    return parser.parse_args()


def training(args):
    start = time.time()

    # load fmri
    fmri = load_nc_fmri(paradigm='all', subject=args.subject, k=5000, mode='mean')

    # load wordvecs
    if args.wordvecs == 'glove':
        wordvecs = scio.loadmat('/home/sxzou/concept_decoding/data/wordvecs/glove_180_words.mat')['glove_v1']
    else:
        wordvec_dir = '/home/sxzou/concept_decoding/data/wordvecs/'
        wordvec_path = os.path.join(wordvec_dir, args.wordvecs + '_180_words.npy')
        wordvecs = np.load(wordvec_path)

    # choose regressor, model training (18-fold cross validation)
    n_splits = 18
    save_dir = os.path.join(args.save_dir, args.subject + '/' + args.wordvecs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.regressor == 'ridge':
        alphas = np.array([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
        if args.encoding:
            kfold_cv_ridge(wordvecs, fmri, n_splits, alphas, save_dir)
        else:
            kfold_cv_ridge(fmri, wordvecs, n_splits, alphas, save_dir)
    else:
        if args.encoding:
            kfold_cv_sklearn_regressor(args, wordvecs, fmri, n_splits, save_dir)
        else:
            kfold_cv_sklearn_regressor(args, fmri, wordvecs, n_splits, save_dir)
    print("Time used {:} (h:mm:ss)".format(format_time(time.time() - start)))


def kfold_cv_ridge(fmri, wordvecs, n_splits, alphas, save_dir):
    '''
    fmri已经归一化，fmri加多1列全1特征，以简化bias的计算
    '''
    n_samples, n_voxels = fmri.shape
    n_targets = wordvecs.shape[1]
    # 在特征矩阵后面加一列全1
    one_column = np.ones((n_samples, 1))
    fmri1 = np.concatenate([fmri, one_column], axis=1)  # feature maxtrix

    valid_predictions = np.zeros((n_samples, n_targets))
    predictions = np.zeros((n_samples, n_targets))

    fold_index_list = word_kfold_split(n_samples, n_splits)
    for k in range(n_splits):
        print('\nFold {}:'.format(k))
        val_index = fold_index_list[k]
        test_index = fold_index_list[(k+1)%n_splits]
        train_index = list(set(np.arange(n_samples)) - set(val_index + test_index))
        train_x = fmri1[train_index]
        train_y = wordvecs[train_index]
        val_x = fmri1[val_index]
        val_y = wordvecs[val_index]
        # search the best alpha
        corrs = ridge_by_lambda(train_x, train_y, val_x, val_y, alphas)  # (n_lambdas, n_targets)
        best_alpha_index = np.argmax(corrs, axis=0)
        # save the correlations
        saved_corr = corrs.max(axis=0)
        save_path = os.path.join(save_dir, 'valid_correlations_fold_' + str(k))
        np.save(save_path, saved_corr)
        print(pd.Series(best_alpha_index).value_counts())

        # # 用train+valid数据重新求解
        # train_x = np.concatenate([train_x, val_x])
        # train_y = np.concatenate([train_y, val_y])

        weights = np.zeros((n_voxels+1, n_targets))
        for idx_alpha in range(len(alphas)):
            # 对相同alpha的回归可以合并求解
            idx_target = (best_alpha_index == idx_alpha)
            weights[:, idx_target] = ridge(train_x, train_y[:, idx_target], alphas[idx_alpha])
        save_path = os.path.join(save_dir, 'reg_weights_fold_' + str(k))
        np.save(save_path, weights)
        print('weights save to ', save_path)
        # predict on the valid data
        valid_predictions[val_index] = np.dot(val_x, weights)
        # predict on the test data
        test_x = fmri1[test_index]
        predictions[test_index] = np.dot(test_x, weights)
    save_path = os.path.join(save_dir, 'valid_predictions')
    np.save(save_path, valid_predictions)
    save_path = os.path.join(save_dir, 'predictions')
    np.save(save_path, predictions)
    print('predictions save to ', save_path)
    print('done')


def single_subject_ridge(subject, wordvecs_type, wordvecs, alphas, k=5000, wv='glove', data_flag='nc', n_splits=10):
    '''
    :param wordvecs: numpy array of shape (nsamples, dim)
    :param alphas: list, the regularization hyparameter in ridge
    :param k: numbers of voxels to keep
    :param wv: what kind of wv used in voxel selection
    '''
    if data_flag == 'nc':
        fmri = load_nc_fmri(subject=subject, k=k, wv=wv)
    else:
        fmri = load_sci_fmri(subject=subject, k=k, wv=wv)

    save_dir = "/home/sxzou/concept_decoding/data/cs_mapping_acl/" + data_flag + "/" + subject + "/" \
               + wordvecs_type + '/' + wv + '_' + str(k)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    kfold_cv_ridge(fmri, wordvecs, n_splits, alphas, save_dir)


def parallel_ridge(wordvecs_type, subjects, data_flag, k, wv):
    start = time.time()

    # load wordvecs
    if data_flag == 'nc':
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

    n_splits = 10
    alphas = np.array([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
    Parallel(n_jobs=8)(delayed(single_subject_ridge)(sub, wordvecs_type, wordvecs, alphas, k, wv, data_flag, n_splits)
                       for sub in subjects)

    print("Time used {:} (h:mm:ss)".format(format_time(time.time() - start)))



if __name__ == '__main__':

    #start = time.time()
    # args = parse_args()
    # print(args)
    # training(args)
    # print("Time used {:} (h:mm:ss)".format(format_time(time.time() - start)))
    #
    # # # model evaluation
    # fmri = load_exp1_fmri(paradigm='all', subject='M15', k=5000, mode='mean')
    # #wordvecs = scio.loadmat('/home/sxzou/concept_decoding/data/wordvecs/glove_180_words.mat')['glove_v1']
    # save_dir = '/home/sxzou/concept_decoding/data/cs_mapping/encoding/M15/ridge/glove'
    # predictions = np.load(os.path.join(save_dir, 'predictions.npy'))
    # pairwise_accuracy(predictions, fmri)
    # topk_accuracy(predictions, fmri)


    DATA_FLAG = 'nc'
    wv = 'bert'

    if DATA_FLAG == 'nc':
        subjects = ["P01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M13", "M14", "M15", "M16", "M17"]
    else:
        subjects = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

    for k in [1000, 5000, 500, 2000, 3000, 4000]:
        for wordvecs_type in ['bert_embed', 'bert_layeravg', 'glove']:
            parallel_ridge(wordvecs_type, subjects, data_flag=DATA_FLAG, k=k, wv=wv)