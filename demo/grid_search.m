
clear all;

% COMPONENT OF START LEARNING SO THAT ONE SCRIPT CAN BE RUN FROM COMMAND LINE
% learnLoc = which('startLearning.m');
% parentDir = fileparts(learnLoc);
% addpath(genpath(parentDir));
% root = parentDir;
% data = fullfile(parentDir,'data');
% modules = fullfile(parentDir,'modules');
% save(fullfile(parentDir,'.config.mat'));


HiddenUnits = 700:-300:100;
learningRates = 0.01:0.03:0.1;
batchSizes = 500:-200:100;
Epochs = 250:-100:100;
Dropouts = 0.3:-0.1:0;

fprintf('\nHere we perform a grid search for RBM hyperparameters with Binary inputs (Silhouette datastet).\n');
sec = length(HiddenUnits)*length(learningRates)*length(batchSizes)*length(Epochs)*40;
fprintf('Estimated time to search = %d hours %d mins',floor(sec/3600.0), floor(mod(sec,3600)/60));

% LOAD DATASET
load('caltech101_silhouettes_28_split1.mat');
test_labels = val_labels;
test_data = val_data;

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

leastError = 100;
bestSetting = struct('Hidden', 100, 'LRate', 0.01, 'BatchSize', 100, 'Epochs', 100, 'Dropout', 0);

for nHid = HiddenUnits
for lRate = learningRates
for batchSize = batchSizes
for epochs = Epochs
for dropout = Dropouts
    
    [nObs,nVis] = size(train_data);

    % DEFINE A MODEL ARCHITECTURE
    arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

    % GLOBAL OPTIONS
    arch.opts = {'verbose', 0, ...
             'lRate', lRate, ...
            'momentum', 0.5, ...
            'nEpoch', epochs, ...
            'wPenalty', 0.02, ...
            'batchSz', batchSize, ...
            'beginAnneal', 10, ...
            'nGibbs', 1, ...
            'sparsity', .01, ...
            'varyEta',7, ...
            'dropout', dropout, ...
            'displayEvery', 20};
    %  		'visFun', @visBinaryRBMLearning};

    % INITIALIZE RBM
    r = rbm(arch);

    % TRAIN THE RBM
    r = r.train(train_data,single(train_labels));

    [~,classErr,misClass] = r.classify(test_data, single(test_labels));
    
    fprintf('For setting hidden = %d, lrate = %f, batchSize = %d, epochs = %d, misclassification error = %f\n',nHid, lRate, batchSize, epochs, classErr*100);
    if classErr*100 < leastError
        leastError = classErr*100
        bestSetting = struct('Hidden', nHid, 'LRate', lRate, 'BatchSize', batchSize, 'Epochs', epochs, 'Dropout', dropout)
    end
end
end
end
end
end