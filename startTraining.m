
learnLoc = which('startLearning.m');
parentDir = fileparts(learnLoc);
% ADD PATHS
addpath(genpath(parentDir));

% SAVE CONSTANTS (SEE medalConstants.m)
root = parentDir;
data = fullfile(parentDir,'data');
modules = fullfile(parentDir,'modules');
save(fullfile(parentDir,'.config.mat'));

% Change this to 1 for using saved model, 0 for learning from scratch
useSavedModel = 0;
silhoutte


