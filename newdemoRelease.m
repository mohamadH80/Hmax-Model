% %demoRelease.m
% %demonstrates how to use C2 standard model features in a pattern classification framework
% 
% addpath ~/scratch2/osusvm/ %put your own path to osusvm here
% 
% useSVM = 0; %if you do not have osusvm installed you can turn this
%             %to 0, so that the classifier would be a NN classifier
% 	    %note: NN is not a great classifier for these features
% 	    
% READPATCHESFROMFILE = 0; %use patches that were already computed
%                          %(e.g., from natural images)
% 
% patchSizes = [4 8 12 16]; %other sizes might be better, maybe not
%                           %all sizes are required
% 			  
% numPatchSizes = length(patchSizes);
% 
% %specify directories for training and testing images
% 
% train_set.pos   = 'C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\Compressed\AnimalDB\AnimalDB\Targets\train';
% train_set.neg   = 'C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\Compressed\AnimalDB\AnimalDB\Distractors\train';
% test_set.pos    = 'C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\Compressed\AnimalDB\AnimalDB\Targets\test';
% test_set.neg    = 'C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\Compressed\AnimalDB\AnimalDB\Distractors\test';
% 
% cI = readAllImages(train_set,test_set); %cI is a cell containing
%                                         %all training and testing images
% 
% if isempty(cI{1}) | isempty(cI{2})
%   error(['No training images were loaded -- did you remember to' ...
% 	' change the path names?']);
% end
%   
% %below the c1 prototypes are extracted from the images/ read from file
% if ~READPATCHESFROMFILE
%   tic
%   numPatchesPerSize = 250; %more will give better results, but will
%                            %take more time to compute
%   cPatches = extractRandC1Patches(cI{1}, numPatchSizes, ...
%       numPatchesPerSize, patchSizes); %fix: extracting from positive only 
%                                       
%   totaltimespectextractingPatches = toc;
% else
%   fprintf('reading patches');
%   cPatches = load('PatchesFromNaturalImages250per4sizes','cPatches');
%   cPatches = cPatches.cPatches;
% end
% 
% %----Settings for Testing --------%
% rot = [90 -45 0 45];
% c1ScaleSS = [1:2:18];
% RF_siz    = [7:2:39];
% c1SpaceSS = [8:2:22];
% minFS     = 7;
% maxFS     = 39;
% div = [4:-.05:3.2];
% Div       = div;
% %--- END Settings for Testing --------%
% 
% fprintf(1,'Initializing gabor filters -- full set...');
% %creates the gabor filters use to extract the S1 layer
% [fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);
% fprintf(1,'done\n');
% 
% %The actual C2 features are computed below for each one of the training/testing directories
% tic
% for i = 1:4,
%   C2res{i} = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cI{i},numPatchSizes);
%   toc
% end
% totaltimespectextractingC2 = toc;
% 
% %Simple classification code
% XTrain = [C2res{1} C2res{2}]; %training examples as columns 
% XTest =  [C2res{3},C2res{4}]; %the labels of the training set
% ytrain = [ones(size(C2res{1},2),1);-ones(size(C2res{2},2),1)];%testing examples as columns
% ytest = [ones(size(C2res{3},2),1);-ones(size(C2res{4},2),1)]; %the true labels of the test set

load('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\XTrain.mat')
load('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\XTest.mat')
load('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\ytrain.mat')
load('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\ytest.mat')
% xtrainb = XTrain(:,1:150); 
% xtrainh = XTrain(:,151:300); 
% xtrainf = XTrain(:,301:450); 
% xtrainm = XTrain(:,451:end); 
xtestb = XTest(:,1:150); 
xtesth = XTest(:,151:300); 
xtestf = XTest(:,301:450); 
xtestm = XTest(:,451:end); 
ytestb = ytest(1:150);
ytesth = ytest(151:300);
ytestf = ytest(301:450);
ytestm = ytest(451:end);


% % Kernels to be used
% kernels = {'polynomial', 'rbf'};
% trainAccuracies = zeros(length(kernels), 1);
% testAccuracies = zeros(length(kernels), 1);
% 
% for i = 1:length(kernels)
%     % Train SVM with different kernels
%     if strcmp(kernels{i}, 'polynomial')
%         svmModel = fitcsvm(XTrain', ytrain', 'KernelFunction', kernels{i}, 'PolynomialOrder', 3);
%     else
%         svmModel = fitcsvm(XTrain', ytrain');
%     end
%     
%     % Predict using the SVM model
%     predictionsTrain = predict(svmModel, XTrain');
%     predictionsTest = predict(svmModel, XTest');
% 
%     % Calculate accuracies
%     trainAccuracies(i) = sum(ytrain == predictionsTrain) / length(ytrain);
%     testAccuracies(i) = sum(ytest == predictionsTest) / length(ytest);
% end
% 
% % Plotting accuracies
% figure;
% plot(1:length(kernels), trainAccuracies, 'b-o', 1:length(kernels), testAccuracies, 'r-o');
% xticks(1:length(kernels));
% xticklabels(kernels);
% xlabel('Kernel Type');
% ylabel('Accuracy');
% legend('Training Accuracy', 'Testing Accuracy');
% title('SVM Training and Testing Accuracies with Different Kernels');
% kernels = {'linear', 'polynomial', 'rbf'};
% trainAccuracies = zeros(length(kernels), 1);
% testAccuracies = zeros(length(kernels), 1);


XTrain = XTrain';
xtestm = xtestm';
xtestf = xtestf';
xtesth = xtesth';
xtestb = xtestb';


% Step 1: Train a linear SVM classifier on the training data
svmModel = fitcsvm(XTrain, ytrain, 'KernelFunction', 'linear');

% Step 2: Split the test data into four groups
frac = 1; % Fraction of data to use for testing
rng(42); % Set seed for reproducibility

xtestb_indices = randperm(size(xtestb, 1), round(frac * size(xtestb, 1)));
xtesth_indices = randperm(size(xtesth, 1), round(frac * size(xtesth, 1)));
xtestm_indices = randperm(size(xtestm, 1), round(frac * size(xtestm, 1)));
xtestf_indices = randperm(size(xtestf, 1), round(frac * size(xtestf, 1)));

xtestb_train = xtestb(setdiff(1:size(xtestb, 1), xtestb_indices), :);
xtestb_test = xtestb(xtestb_indices, :);


xtesth_train = xtesth(setdiff(1:size(xtesth, 1), xtesth_indices), :);
xtesth_test = xtesth(xtesth_indices, :);

xtestm_train = xtestm(setdiff(1:size(xtestm, 1), xtestm_indices), :);
xtestm_test = xtestm(xtestm_indices, :);

xtestf_train = xtestf(setdiff(1:size(xtestf, 1), xtestf_indices), :);
xtestf_test = xtestf(xtestf_indices, :);

% Step 3: Use the trained SVM classifier to predict labels for each group
y_pred_xtestb = predict(svmModel, xtestb_test);
y_pred_xtesth = predict(svmModel, xtesth_test);
y_pred_xtestm = predict(svmModel, xtestm_test);
y_pred_xtestf = predict(svmModel, xtestf_test);

% Step 4: Evaluate the accuracy of the predictions for each group
accuracy_xtestb = sum(y_pred_xtestb == ytestb) / length(ytestb);
accuracy_xtesth = sum(y_pred_xtesth == ytesth) / length(ytesth);
accuracy_xtestm = sum(y_pred_xtestm == ytestm) / length(ytestm);
accuracy_xtestf = sum(y_pred_xtestf == ytestf) / length(ytestf);

% Assuming you have calculated accuracies as mentioned in the previous response
% 
% % Bar plot
% figure;
% 
% bar([accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100);
% 
% title('Accuracy for Each Group');
% xlabel('Groups');
% ylabel('Accuracy (%)');
% xticklabels({'xtestb', 'xtesth', 'xtestm', 'xtestf'});
% ylim([0, 100]);
% 
% % Display the values on top of each bar
% text(1:length([accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf]), ...
%      [accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100, ...
%      sprintfc('%.2f%%', [accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100), ...
%      'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% 
% grid on;
% 

% Line plot
figure;

% Assuming accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf are defined
% Plotting the accuracy for each group
plot([accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100, '-o');

title('Accuracy for Each Group');
xlabel('Groups');
ylabel('Accuracy (%)');
xticklabels({'xtestb', 'xtesth', 'xtestm', 'xtestf'});
xticks(1:4); % Set x-axis ticks to match the number of groups
ylim([0, 100]);

% Display the values on top of each point
text(1:length([accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf]), ...
     [accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100, ...
     sprintfc('%.2f%%', [accuracy_xtestb, accuracy_xtesth, accuracy_xtestm, accuracy_xtestf] * 100), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

grid on;

