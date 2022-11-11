import os
import numpy as np
import scipy.io as scio   #读取mat文件

# Z-score -- z-score each column
zscore = lambda v: (v-v.mean(0))/v.std(0)


def preprocess_sci_fmri_data(subject, words):
    '''
    preprocess fmri data, average across 6 trials to get the fri for each word
    :param subject: P1-P9
    :param words: 60 stimuli words in specific order
    :return: numpy array of shape (60, nvoxels)
    '''
    path = '/home/sxzou/concept_decoding/Science_data/data-science-' + subject
    data = scio.loadmat(path)
    # get word
    trail_words = data['info']['word'].squeeze()   #(360,)
    trail2word = [[] for i in range(60)]  # 存储对应于每个词语的trail,用于提取相应的fMRI
    for i in range(len(trail_words)):
        w = trail_words[i][0]
        ind = words.index(w)
        trail2word[ind].append(i)
    # get fmri data
    nvoxels = data['meta']['nvoxels'][0,0][0,0]
    fmri_trails = data['data'].squeeze()  #(360,)
    fmri = np.zeros((60, nvoxels))
    for i, w in enumerate(words):
        ind_list = trail2word[i]
        assert len(ind_list) == 6, 'number of trails is not correct!'
        for ind in ind_list:
            fmri[i] += fmri_trails[ind].squeeze()
    fmri /= 6
    return fmri


def select_informative_voxel_index(subject='M15', k=5000, wv='glove', mode='mean', data_flag='nc'):
    '''
    vs_scores.mat：(300, #voxels), informative score
    wv: word vectors that used in computing informative scores, "glove" or "bert"
    '''
    if data_flag == 'nc':
        file_dir = '/home/sxzou/concept_decoding/NC_data/informative_score/' + wv
    else:
        file_dir = '/home/sxzou/concept_decoding/Science_data/preprocessed_fmri/informative_score/' + wv
    vs_path = os.path.join(file_dir, subject + '_vs_scores.mat')
    scores = scio.loadmat(vs_path)['scores']   # (300, 185703) or (300, 21764)
    if mode == 'max':
        scores = scores.max(axis=0)
    else:
        scores = scores.mean(axis=0)
    top_k_index = np.argsort(scores)[-k:]
    assert len(top_k_index) == k, 'voxels numbers is wrong'
    return top_k_index


def load_nc_fmri(paradigm='all', subject='M15', k=5000, wv='glove', mode='mean'):
    '''
    :param paradigm: sent / picture / wordcloud / all
    '''
    data_dir = '/home/sxzou/concept_decoding/NC_data'
    if paradigm == 'sent':
        path = os.path.join(data_dir, subject + '/data_180concepts_sentences.mat')
        fmri = scio.loadmat(path)['examples']
    elif paradigm == 'picture':
        path = os.path.join(data_dir, subject + '/data_180concepts_pictures.mat')
        fmri = scio.loadmat(path)['examples']
    elif paradigm == 'wordcloud':
        path = os.path.join(data_dir, subject + '/data_180concepts_wordclouds.mat')
        fmri = scio.loadmat(path)['examples']
    else:
        # 三种范式平均
        path = os.path.join(data_dir, subject + '/data_180concepts_sentences.mat')
        fmri = scio.loadmat(path)['examples']
        path = os.path.join(data_dir, subject + '/data_180concepts_pictures.mat')
        fmri += scio.loadmat(path)['examples']
        path = os.path.join(data_dir, subject + '/data_180concepts_wordclouds.mat')
        fmri += scio.loadmat(path)['examples']
        fmri = fmri / 3
    # select informative voxels
    top_k_index = select_informative_voxel_index(subject, k, wv, mode, data_flag='nc')
    fmri = fmri[:, top_k_index]
    assert fmri.shape == (180, k), 'fmri shape error'
    return fmri


def load_sci_fmri(subject, k=5000, wv='glove', mode="mean"):
    '''
    :param subject: P1-P9
    :param k: how many voxels to keep
    :param wv: word vectors that used in computing informative scores, "glove" or "bert"
    :param mode: default mean
    :return: reduced fmri
    '''
    fmri_path = '/home/sxzou/concept_decoding/Science_data/preprocessed_fmri/' + subject + '.mat'
    data = scio.loadmat(fmri_path)['fmri']    #(60, 21764)
    # select informative voxels
    top_k_index = select_informative_voxel_index(subject, k, wv, mode, data_flag='sci')
    fmri = data[:, top_k_index]
    assert fmri.shape == (60, k), 'fmri shape error'
    return fmri


def get_nc_fmri_words(subject='M15'):
    data_dir = '/home/sxzou/concept_decoding/NC_data'
    path = os.path.join(data_dir, subject + '/data_180concepts_sentences.mat')
    data = scio.loadmat(path)
    words = data['keyConcept'].squeeze()
    words = [words[i][0] for i in range(len(words))]
    # 'counting' should be corrected as 'count' for word
    words[words.index('counting')] = 'count'
    # get pos tagging: 1: noun, 2: verb, 3: adjective, 4: adverb
    pos_tag = data['labelsPOS'].squeeze()
    # get concretness, 介于0-5, 值越大表示越具象
    concreteness = data['labelsConcreteness'].squeeze()/5
    return words, pos_tag, concreteness





if __name__ == '__main__':

    # from text_preprocessing import load_text
    # words = load_text('/home/sxzou/concept_decoding/Science_data/word_stimuli.txt')

    # for subject in range(1,10):
    #     print(subject)
    #     fmri = preprocess_fmri_data(subject, words)
    #     save_path = '/home/sxzou/concept_decoding/Science_data/P' + str(subject) + '_fmri.npy'
    #     np.save(save_path, fmri)

    # embed = get_glove_embed(words)
    # save_path = '/home/sxzou/concept_decoding/Science_data/glove_embed.npy'
    # np.save(save_path,  embed)

    # 将npy转换为mat格式
    path = '/home/sxzou/concept_decoding/data/wordvecs/nc/bert_layeravg_180_words.npy'
    data = np.load(path)
    mat_path = '/home/sxzou/concept_decoding/NC_data/bert_layeravg_180.mat'
    scio.savemat(mat_path, {'bert': data})

    # for subject in range(1,10):
    #     print(subject)
    #     path = '/home/sxzou/concept_decoding/Science_data/P' + str(subject) + '_fmri.npy'
    #     data = np.load(path)
    #     mat_path = '/home/sxzou/concept_decoding/Science_data/preprocessed_fmri/P' + str(subject) + '.mat'
    #     scio.savemat(mat_path, {'fmri': data})


