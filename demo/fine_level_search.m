
clear all;

% COMPONENT OF START LEARNING SO THAT ONE SCRIPT CAN BE RUN FROM COMMAND LINE
% learnLoc = which('startLearning.m');
% parentDir = fileparts(learnLoc);
% addpath(genpath(parentDir));
% root = parentDir;
% data = fullfile(parentDir,'data');
% modules = fullfile(parentDir,'modules');
% save(fullfile(parentDir,'.config.mat'));

N = 5;
M = 4;

HiddenUnits = [400,700];
learningRates = [0.05, 0.1];
momentums = [0.4, 0.7];
batchSizes = [150,200];
Epochs = [200,300];
Dropouts = [0,0.1];
wdecays = [0,0.01];


fprintf('\nHere we perform a fine level search for RBM hyperparameters with Binary inputs (Silhouette datastet).\n');

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

leastError = 30.3445;

for hyperParameterType = 1:7
for division = 1:(N+1)    
for randomSample = 1:(M)   
    
    % Randomly sample all values from a uniform distribution
    nHid = randi(HiddenUnits);
    lRate = learningRates(1) + rand(1)*(learningRates(2) - learningRates(1));
    batchSize = randi(batchSizes);
    epochs = randi(Epochs);
    dropout = Dropouts(1) + rand(1)*(Dropouts(2) - Dropouts(1)); 
    momentum = momentums(1) + rand(1)*(momentums(2) - momentums(1)); 
    wdecay = wdecays(1) + rand(1)*(wdecays(2) - wdecays(1)); 
    
    switch hyperParameterType
       case 1 
           nHid = HiddenUnits(1) + (division-1)*(HiddenUnits(2) - HiddenUnits(1))/N;
        case 2
            lRate = learningRates(1) + (division-1)*(learningRates(2) - learningRates(1))/N;
        case 3
            batchSize = batchSizes(1) + (division-1)*(batchSizes(2) - batchSizes(1))/N;
        case 4
            epochs = Epochs(1) + (division-1)*(Epochs(2) - Epochs(1))/N;
        case 5
            dropout = Dropouts(1) + (division-1)*(Dropouts(2) - Dropouts(1))/N;
        case 6
            momentum = momentums(1) + (division-1)*(momentums(2) - momentums(1))/N;
        case 7
            wdecay = wdecays(1) + (division-1)*(wdecays(2) - wdecays(1))/N;
    end
    
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
            'momentum', momentum, ...
            'wPenalty', wdecay, ...
            'displayEvery', 20};
    %  		'visFun', @visBinaryRBMLearning};

    % INITIALIZE RBM
    r = rbm(arch);

    % TRAIN THE RBM
    r = r.train(train_data,single(train_labels));

    [~,classErr,misClass] = r.classify(test_data, single(test_labels));
    
    fprintf('For setting type = %d, hidden = %d, lrate = %f, batchSize = %d, epochs = %d, Dropout = %f, momentum = %f, wdecay = %f, misclassification error = %f\n',hyperParameterType, nHid, lRate, batchSize, epochs, dropout, momentum, wdecay, classErr*100);
    if classErr*100 < leastError
        leastError = classErr*100
    end
end
end
end