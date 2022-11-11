% perform voxel seletection for every subject of fMRI60, the data and the code are in the same folder

subjects = ['P1', 'P2', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'];

% word embedding for the 60 stimuli
load('bert_layeravg.mat');

for i = 1:length(subjects)
    sub = char(subjects(i));
    disp(['Start computing ', sub]);
    % load fmri data
    load([sub, '.mat']);
    scores = trainVoxelwiseTargetPredictionModels(fmri, wordvec);
    save(['informative_score/', sub, '_vs_scores.mat'], 'scores');
end
