

% perform voxel seletection for every subject

% array
%subjects = ["M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M13", "M14", "M15", "M16", "M17"];
subjects = ["P01"];

%load('glove_v1.mat');
%load('bert_layeravg_180.mat');
load('bart_encoder_180_words.mat');

for i = 1:length(subjects)
    sub = char(subjects(i));
    disp(['Start computing ', sub]);
    
    load([sub, '/data_180concepts_pictures.mat']);
    picture_fmri = examples;

    load([sub, '/data_180concepts_sentences.mat']);
    sent_fmri = examples;

    load([sub, '/data_180concepts_wordclouds.mat']);
    cloud_fmri = examples;

    fmri = (picture_fmri + sent_fmri + cloud_fmri) / 3;

    %scores = trainVoxelwiseTargetPredictionModels(fmri, glove_v1, 'meta', meta);
    %scores = trainVoxelwiseTargetPredictionModels(fmri, bert, 'meta', meta);
    scores = trainVoxelwiseTargetPredictionModels(fmri, bart_encoder, 'meta', meta);

    save(['informative_score/bart_encoder/', sub, '_vs_scores.mat'], 'scores');
    
    %clearvars except glove_v1 subjects i;

end