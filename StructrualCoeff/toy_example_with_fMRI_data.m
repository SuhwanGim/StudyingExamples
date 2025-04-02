clc; clear;
%% addpath
addpath(genpath('/Users/suhwan/Dropbox/github/CanlabTools'))
addpath(genpath('/Users/suhwan/Dropbox/github/CocoanTools/cocoanCORE'))
addpath(genpath('/Users/suhwan/Dropbox/github/external_toolbox/spm25'))
addpath(genpath('/Users/suhwan/Dropbox/github/external_toolbox/gramm'))
rmpath(genpath('/Users/suhwan/Dropbox/github/external_toolbox/spm25/external/'));
rmpath(genpath('/Users/suhwan/Dropbox/github/external_toolbox/spm25/external/fieldtrip/compat'));
%% Load data 
basedir = '/Users/suhwan/Dropbox/github/suhwan_github';
addpath(basedir)
fmri_data_file = which('bmrk3_6levels_pain_dataset.mat');

% if isempty(fmri_data_file)
% 
%     % attempt to download
%     disp('Did not find data locally...downloading data file from figshare.com')
% 
%     fmri_data_file = websave('bmrk3_6levels_pain_dataset.mat', 'https://ndownloader.figshare.com/files/12708989');
%     disp('done')
% end
%
load(fmri_data_file);

descriptives(image_obj);
%% Collect some variables we need

% subject_id is useful for cross-validation
%
% this used as a custom holdout set in fmri_data.predict() below will
% implement leave-one-subject-out cross-validation

subject_id = image_obj.additional_info.subject_id;

% ratings: reconstruct a subjects x temperatures matrix
% so we can plot it
%
% The command below does this only because we have exactly 6 conditions
% nested within 33 subjects and no missing data.

ratings = reshape(image_obj.Y, 6, 33)';

% temperatures are 44 - 49 (actually 44.3 - 49.3) in order for each person.

temperatures = image_obj.additional_info.temperatures;

% plot results
create_figure('ratings');
lineplot_columns(ratings, 'color', [.7 .3 .3], 'markerfacecolor', [1 .5 0]);
xlabel('Temperature');
ylabel('Rating');

%% Prediction 
% relevant functions:
% predict method (fmri_data)
% predict_test_suite method (fmri_data)

% Define custom holdout set.  If we use subject_id, which is a vector of
% integers with a unique integer per subject, then we are doing
% leave-one-subject-out cross-validation.

% let's build five-fold cross-validation set that leaves out ALL the images
% from a subject together. That way, we are always predicting out-of-sample
% (new individuals). If not, dependence across images from the same
% subjects may invalidate the estimate of predictive accuracy.

holdout_set = zeros(size(subject_id));      % holdout set membership for each image
n_subjects = length(unique(subject_id));
C = cvpartition(n_subjects, 'KFold', 10);
for i = 1:5
    teidx = test(C, i);                     % which subjects to leave out
    imgidx = ismember(subject_id, find(teidx)); % all images for these subjects
    holdout_set(imgidx) = i;
end

algoname = 'cv_lassopcr'; % cross-validated penalized regression. Predict pain ratings

[cverr, stats, optout] = predict(image_obj, 'algorithm_name', algoname, 'nfolds', holdout_set,'bootsamples',1000);
save('StructrualCoeff/toy_example_fMRI_predict_resutls_boot1000.mat','stats');
%[cverr, stats, optout] = predict(image_obj, 'algorithm_name', algoname, 'nfolds', holdout_set);

%% Structure coefficient
load('StructrualCoeff/toy_example_fMRI_predict_resutls_boot1000.mat','stats');
nVOX = size(stats.weight_obj.dat,1);
SC_dat = NaN(nVOX,1);
SC_p =  NaN(nVOX,1);
for vox_i = 1:nVOX 
    [SC_dat(vox_i,1), SC_p(vox_i,1)] = corr(image_obj.dat(vox_i,:)', stats.yfit);
    % disp(vox_i);
end
%% construct map 
temp_SC_map = image_obj; %structure coefficient map 
temp_SC_map.dat = SC_dat.* double(SC_p < 0.005);


temp_W_map = image_obj; %weight map
temp_W_map.dat = stats.WTS.wmean'.*(stats.WTS.wP' < 0.005);

%% Visualization 
orthviews_multiple_objs({temp_SC_map temp_W_map})


%%

pm_obj = predictive_model(stats);

%%
% help xval_regression_multisubject
% help xval_lasso_brain