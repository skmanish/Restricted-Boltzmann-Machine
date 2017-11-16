
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

nHid = 610; % 610 HIDDEN UNITS

% DEFINE A MODEL ARCHITECTURE
arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

% GLOBAL OPTIONS
arch.opts = {'verbose', 1, ...
		 'lRate', 0.094360, ...
		'momentum', 0.5, ...
		'nEpoch', 243, ...
		'wPenalty', 0.002897, ...
		'batchSz', 179, ...
		'beginAnneal', 10, ...
		'nGibbs', 1, ...
		'sparsity', .01, ...
		'varyEta',7, ...
        'dropout', 0.064756, ...
        'momentum', 0.559208, ...
        'wPenalty', 0.002, ...
		'displayEvery', 20};
%  		'visFun', @visBinaryRBMLearning};

if(useSavedModel==0)
        % INITIALIZE RBM
        r = rbm(arch);
        % TRAIN THE RBM
        r = r.train(train_data,single(train_labels));
    else 
        load('r')
end
[~,classErr,misClass] = r.classify(test_data, single(test_labels));


fprintf("Misclassification error on testing set = %.2f\n", classErr*100);