import scipy
from scipy.stats import ttest_rel
from statsmodels.stats import multitest
import pandas as pd
import numpy as np
import os



if __name__ == '__main__':

    DATA_FLAG = 'nc'
    # test_names = ['direct_bert_embed_alpha0.7',
    #              'retri_bert_layeravg_k5_alpha0.7',
    #              'random_retri_bert_layeravg_k5_alpha0.7']
    test_names = ['retri_bert_layeravg_k5_alpha0.7']



    if DATA_FLAG == 'nc':
        subjects = ["P01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M13", "M14", "M15", "M16",
                    "M17"]
    else:
        subjects = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

    data_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/cloze/'+DATA_FLAG
    baseline = pd.read_csv(os.path.join(data_dir, 'baseline.csv')).values   # (4,10)

    save_dir = '/home/sxzou/concept_decoding/data/accuracy_acl/final/cloze'
    for test_name in test_names:
        print(test_name)
        ours = pd.read_csv(os.path.join(save_dir, DATA_FLAG +'/fusion/'+test_name+'.csv')).values  #(n_subjects*4, 10)
        test_pvalue = np.zeros((len(subjects), 4))
        for i, sub in enumerate(subjects):
            pv = list(ttest_rel(baseline, ours[i * 4: (i + 1) * 4], axis=1, alternative='less').pvalue)
            test_pvalue[i] = pv
            print(sub, pv)
        # save_path = os.path.join(save_dir, 'significance/'+ DATA_FLAG +'/'+test_name+'.csv')
        # test_pvalue = pd.DataFrame(test_pvalue)
        # test_pvalue.to_csv(save_path, index=False)

        # FDR校正
        q = multitest.fdrcorrection(test_pvalue[:, 1], alpha=0.1, method='indep', is_sorted=False)
        print(q)