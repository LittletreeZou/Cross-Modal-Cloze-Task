

% perform voxel seletection for every subject

% array
subjects = ['s1_tianke', 's2_chenzhonghao', 's3_huangxin', 's4_luyu', 's5_liwenlu', 's6_zhangruotong',
            's7_heqin', 's8_chenguohua', 's9_maliqun', 's10_xuweilin', 's11_zhushanshan'];

load('tencent_wvs.mat');

for i = 1:length(subjects)
    sub = char(subjects(i));
    disp(['Start computing ', sub]);
    
    load([sub, '.mat']);

    scores = trainVoxelwiseTargetPredictionModels(fmri, wordvec);
    
    save(['informative_score/', sub, '_vs_scores.mat'], 'scores');
    

end