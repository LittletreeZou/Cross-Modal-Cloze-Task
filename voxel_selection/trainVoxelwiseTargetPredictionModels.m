% Code taken from https://osf.io/crwz7/

% Score voxels by how well a regression model trained on the voxel
% (+neighbours) predicts each column of a target matrix.
%
% Input:
% - examples - #examples x #voxels
% - target   - #examples x #dimensionsTarget
%
% Optional input:
% - 'groupby',<groups for crossvalidation> (defaults to 10-fold)
% - 'voxelMask',<binary mask to exclude voxels of no interest>
% - 'meta',<meta> - use ridge regression from searchlight (voxel + 3D neighbours)
% - 'lambda',<lambda for ridge regression> (default is 1)
%
% Output:
% - predictions - #targets x #voxels
%   per voxel, correlation between target and cross-validated prediction
%
% Examples:
%
%   [scores] = ...
%     trainVoxelwiseTargetPredictionModels(examples,targets,'predictTargetsFromVoxel','lambda',1,'meta',meta);
%
% Notes:
% - email any questions to francisco.pereira@gmail.com
% - code written by Francisco Pereira, Max Wong, and Charles Zheng
%

function [scores] = trainVoxelwiseTargetPredictionModels(varargin)

%
% process parameters
%

if nargin < 2
    fprintf('syntax: trainVoxelwiseTargetPredictionModes(<examples>,<targets>,\[optional\])\n');return;
end

examples = varargin{1}; [n2,m]  = size(examples);
targets  = varargin{2}; [n1,mt] = size(targets);
if n1 ~= n2
    fprintf('error: targets must have as many rows as examples\n'); return;
end
n = n1;

% defaults
meta = [];
lambda = 1;
useCorrelation = 0;
useOptimization= 0;
voxelMask = ones(1,m);
% 10-fold
labelsGroup = 1+rem((1:n)',10);

idx = 3;
while idx <= nargin
    argval = varargin{idx}; idx = idx + 1;
    switch argval
      case {'meta'}
        % ridge regression with searchlight
        meta = varargin{idx}; idx = idx + 1;
      case {'groupby'}
        % ridge regression with groups for leave-one-group out cross-validation
        labelsGroup = varargin{idx}; idx = idx + 1;
      case {'lambda'}
        % ridge regression
        lambda = varargin{idx}; idx = idx + 1;
      case {'voxelMask'}
        % mask to restrict the voxels considered (out of all in meta)
        voxelMask = varargin{idx}; idx = idx + 1;
      case {'useOptimization'}
        % various optimizations (experimental, turned off by default)
        useOptimization = varargin{idx}; idx = idx + 1;
      otherwise
        fprintf('error: unknown parameter %s\n',argval);return;
    end
end

%
% main loop
%

useSearchlight = ~isempty(meta);

%% precompute a few things

onecol    = ones(n,1); % useful later
targetsZ  = zscore(targets); % z-scored version to use in regression (and correlation computation)
targetsC  = targets - repmat(mean(targets,1),n,1);
examplesZ = examples - repmat(mean(examples,1),n,1);
predicted = zeros(n,mt); % stores predictions of each target for a voxel

% #targets x #voxel
scores = zeros(mt,m); 

% groups to use in cross-validation, if required
groups = unique(labelsGroup); nGroups = length(groups);
for ig = 1:nGroups
    mask = (labelsGroup == groups(ig));
    indicesTest{ig}  = find( mask); nTest(ig)  = length(indicesTest{ig});
    indicesTrain{ig} = find(~mask); nTrain(ig) = length(indicesTrain{ig});
    targetsPerGroup{ig} = targets(indicesTrain{ig},:);
end

%fprintf('using ');
%if useSearchlight
%    fprintf('ridge regression with lambda=%f ',lambda);
%else
%    fprintf('vanilla regression ');
%end
%fprintf('in cross-validation with %d folds\n',nGroups);

%% voxelwise loop
%% (cross-validation is run for each voxel + searchlight (optionally))

nTargets = size(targetsC,2);
if useOptimization == 1
    xTx = zeros(nGroups,27,27);
    xTy = zeros(nGroups,27,nTargets);
end

% split data into folds prior to looping through voxels
if useOptimization == 3 || useOptimization == 4
    for ig = 1:nGroups
        data{ig} = examplesZ(labelsGroup == ig,:);
        targetsCFolds{ig} = targetsC(labelsGroup == ig, :);
    end
end
if useOptimization == 4
    AxTx = zeros(27*27*(nGroups+1),1);
    BxTy = zeros(27*nTargets*(nGroups+1),1);
end

for v = 1:m
    if rem(v,1000) == 0
        fprintf('iter: %d\n',v)
    end
    if voxelMask(v)

        if useSearchlight
            %% find the neighbours of each voxel and assemble a matrix with
            % - value of the voxels across all examples
            % - values of the neighbours across all examples
            % - column of ones (to use in regression)
            nn = meta.numberOfNeighbours(v);
            nn1 = nn+1;
            neighbours = [v,meta.voxelsToNeighbours(v,1:nn)];
            regularizationMatrix = lambda*eye(nn1);
        else
            % neighbourhood is the voxel by itself
            nn = 0;
            neighbours = v;
            regularizationMatrix = lambda;
        end

        if useOptimization == 1
            data = examplesZ(:,neighbours);
            for ig = 1:nGroups
                xTmp = data(indicesTest{ig},:);     % data for single group
                yTmp = targetsC(indicesTest{ig},:); % targets for single group
                xTx(ig,1:nn1,1:nn1) = xTmp'*xTmp;  % X'gXg
                xTy(ig,1:nn1,:) = xTmp'*yTmp;      % X'gYg
            end
            xTxTmp = xTx(:,1:nn1,1:nn1);
            xTyTmp = xTy(:,1:nn1,:);
            A = reshape(sum(xTxTmp,1),[nn1,nn1]);      % get rid of singleton dimension
            B = reshape(sum(xTyTmp,1),[nn1,nTargets]); % get rid of singleton dimension
            % A = squeeze(sum(xTxTmp,1));              % massively slower than reshape
            % B = squeeze(sum(xTyTmp,1));              % massively slower than reshape
            for ig = 1:nGroups
                predicted(indicesTest{ig},:) = data(indicesTest{ig},:) * ...
                    ((A-reshape(xTxTmp(ig,:,:),[nn1,nn1])+regularizationMatrix) \ ...
                    (B-reshape(xTyTmp(ig,:,:),[nn1,nTargets])));
            end
        elseif useOptimization == 2
            data = examplesZ(:,neighbours);
            for ig = 1:nGroups
                xTmp = data(indicesTest{ig},:);     % data for single group
                yTmp = targetsC(indicesTest{ig},:); % targets for single group
                xTmpT = xTmp';
                xTx{ig} = xTmpT*xTmp;      % X'gXg
                xTy{ig} = xTmpT*yTmp;      % X'gYg
            end
            A = xTx{1};
            B = xTy{1};
            for ig = 2:nGroups
                A = A + xTx{ig};
                B = B + xTy{ig};
            end
            for ig = 1:nGroups
                predicted(indicesTest{ig},:) = data(indicesTest{ig},:) * ...
                    ((A-xTx{ig}+regularizationMatrix) \ ...
                    (B-xTy{ig}));
            end
        elseif useOptimization == 3
            for ig = 1:nGroups
                xTmp = data{ig}(:,neighbours);
                yTmp = targetsCFolds{ig};
                xTmpT = xTmp';
                xTx{ig} = xTmpT*xTmp;      % X'gXg
                xTy{ig} = xTmpT*yTmp;      % X'gYg
            end
            A = xTx{1};
            for ig = 2:nGroups
                A = A + xTx{ig};
            end
            B = xTy{1};
            for ig = 2:nGroups
                B = B + xTy{ig};
            end
            if v == 0; keyboard; end
            for ig = 1:nGroups
                tmp = data{ig}(:,neighbours);
                betas = (A - xTx{ig} + regularizationMatrix) \ (B - xTy{ig});
                predicted(indicesTest{ig},:) = data{ig}(:,neighbours)*betas;
                %predicted(indicesTest{ig},:) = data{ig}(:,neighbours) * ...
                %    ((A-xTx{ig} + regularizationMatrix) \ ...
                %    (B-xTy{ig}));
            end
        elseif useOptimization == 4
            for ig = 1:nGroups
                dTmp{ig} = data{ig}(:,neighbours);
            end
            mexMatrixProducts(dTmp, targetsCFolds, AxTx, BxTy);
            AxTxTmp = reshape(AxTx(1:nn1*nn1*(nGroups+1)),[nn1,nn1,nGroups+1]);
            AxTxTmp(:,:,1) = AxTxTmp(:,:,1) + regularizationMatrix;
            BxTyTmp = reshape(BxTy(1:nn1*nTargets*(nGroups+1)),[nn1,nTargets,nGroups+1]);
            if v == 0; keyboard; end
            for ig = 1:nGroups
                predicted(indicesTest{ig},:) = dTmp{ig} * ( ...
                    (AxTxTmp(:,:,1) - AxTxTmp(:,:,ig+1)) \ ...
                    (BxTyTmp(:,:,1) - BxTyTmp(:,:,ig+1)));
            end
        else
            data = examplesZ(:,neighbours);
            % run cross-validation loop
            for ig = 1:nGroups
                tmp = data(indicesTrain{ig},:);
                % more numerically stable version of the normal equations
                betas = (tmp'*tmp + regularizationMatrix)\(tmp'*targetsC(indicesTrain{ig},:));

                predicted(indicesTest{ig},:) = data(indicesTest{ig},:)*betas;
            end
        end
        scores(:,v) = sum(targetsZ .* zscore(predicted),1)/(n-1);
    end; % loop over voxel mask
end; % for loop over voxels
