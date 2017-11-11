clear all;
fprintf('\nHere we train an RBM with Binary inputs (Silhouette datastet).\n');

% LOAD DATASET
load('caltech101_silhouettes_28_split1.mat');

% REASSIGN LABELS INTO A ONE-HOT VECTOR SCHEMA
temp1 = zeros(length(train_labels), max(train_labels));
temp2 = zeros(length(test_labels), max(train_labels));
for i = 1:length(train_labels)
    temp1(i, train_labels(i)) = 1;
end
for i = 1:length(test_labels)
    temp2(i, test_labels(i)) = 1;
end
train_labels = temp1;
test_labels = temp2;
clear temp1, temp2; 

[nObs,nVis] = size(train_data);

nHid = 500; % 500 HIDDEN UNITS

% DEFINE A MODEL ARCHITECTURE
arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

% GLOBAL OPTIONS
arch.opts = {'verbose', 1, ...
		 'lRate', 0.1, ...
		'momentum', 0.5, ...
		'nEpoch', 10, ...
		'wPenalty', 0.02, ...
		'batchSz', 100, ...
		'beginAnneal', 10, ...
		'nGibbs', 1, ...
		'sparsity', .01, ...
		'varyEta',7, ...
		'displayEvery', 20};
%  		'visFun', @visBinaryRBMLearning};

% INITIALIZE RBM
r = rbm(arch);

% TRAIN THE RBM
r = r.train(train_data,single(train_labels));

[~,classErr,misClass] = r.classify(test_data, single(test_labels));


misClass = test_data(misClass,:);
clf; visWeights(misClass',0,[0 1]); title(sprintf('Missclassified -- Error=%1.2f %%',classErr*100));

nVis = 100;
figure; visWeights(r.W(:,1:nVis));
title('Sample of RBM Features');