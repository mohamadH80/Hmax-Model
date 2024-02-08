% Define experiment parameters
numImagesPerFolder = 60;
numFolders = 2;
numBlocks = 10;
block_number = 10;
numTrials = 120;  % Total number of trials (adjust based on your requirements)
imageDuration = 0.02; % 20 ms for each image
blankDuration = 0; % 30 ms blank screen
maskDuration = 0.08;  % 80 ms for the mask
% responseDuration = 1;   % Time allowed for response (in seconds)

% Set up Psychtoolbox
Screen('Preference', 'SkipSyncTests', 1);
screenNumber = max(Screen('Screens'));
% gray = GrayIndex(w);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, [127 127 127]);

fixCrossDimPix = 10; % Size of the arms of the fixation cross in pixels
white = WhiteIndex(screenNumber); % Define white using the screen's white color
black = BlackIndex(screenNumber);

% Generate a random order for presenting images
imageOrder = randperm(numTrials);

maskImage = imread('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\human\mask.png');  % Replace with the actual path
maskTexture = Screen('MakeTexture', window, maskImage);

% Initialize variables to store accuracy and reaction time for each trial
accuracyArray = zeros(4, numTrials/4);
reactionTimeArray = zeros(1, numTrials);

if block_number == 1
    subjectAccuracy = zeros(5, numBlocks);  % Now 5 rows to include reaction time
    save('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\human\results\subject1\subjectAccuracy.mat','subjectAccuracy');
end
load('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\human\results\subject1\subjectAccuracy.mat')

pth = ['C:/Users/USER/Desktop/darsi/courses/FoNSc/pnas_package/DS/human/block',num2str(block_number),'/%d'];

r1 = 1;
r2 = 1;
r3 = 1;
r4 = 1;

% Run the experiment
for trial = 1:numTrials
    % Determine folder and image index based on random order
    folder = mod(imageOrder(trial) - 1, numFolders) + 1;
    imageIndex = ceil(imageOrder(trial) / numFolders);

    % Load image
    folderPath = sprintf(pth, folder);
    images = dir(fullfile(folderPath, '*.jpg'));  
    imagePath = fullfile(folderPath, images(imageIndex).name);
    image = imread(imagePath);
    image = rgb2gray(image);
    % Step 1: Read the Image
    img = im2gray(image); % Convert to grayscale if it's not already
    [rows, cols] = size(img);
    flattened_img = reshape(img, 1, []);
    shuffled_indices = randperm(length(flattened_img));
    shuffled_img = flattened_img(shuffled_indices);
    shuffled_img = reshape(shuffled_img, rows, cols);



    % Calculate the coordinates for the fixation cross
    [xCenter, yCenter] = RectCenter(windowRect);
    xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
    yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
    allCoords = [xCoords; yCoords];
    Screen('DrawLines', window, allCoords, 2, white, [xCenter, yCenter]);
    Screen('Flip', window);
    WaitSecs(0.5);


    % Display image
    imageTexture = Screen('MakeTexture', window, image);
    Screen('DrawTexture', window, imageTexture);
    [~, on_time] = Screen('Flip', window);
    WaitSecs(imageDuration);

    % Display blank screen
    [~, off_time] = Screen('Flip', window);
    WaitSecs(blankDuration);
     mamad = off_time - on_time;

    % Display mask image
    maskTexture = Screen('MakeTexture', window, shuffled_img);
    Screen('DrawTexture', window, maskTexture);
    Screen('Flip', window);
    WaitSecs(maskDuration);

%     % Display blank screen
%     Screen('Flip', window);
%     WaitSecs(blankDuration);
    Screen('DrawLines', window, allCoords, 2, black, [xCenter, yCenter]);
    Screen('Flip', window);
%     WaitSecs(0.5);

    % Wait for user response
    startTime = GetSecs;
    response = [];
    reactionTime = NaN; % Initialize reaction time as NaN
    keyIsDown = 0;
    while ~keyIsDown
        [keyIsDown, secs, keyCode] = KbCheck;
        if keyIsDown % Record the first key press only
            reactionTime = secs - startTime; % Record reaction time
            if keyCode(39)==1
                response = 2;
            elseif keyCode(37) == 1
                response = 1;  
            end
            break;
        end
    end

    % Store reaction time and accuracy for the current trial
    reactionTimeArray(trial) = reactionTime;
    if isempty(response)
        accuracy = NaN;
    else
        accuracy = (response == folder);
    end
    ind = 74 + (block_number==10);
    if imagePath(ind) == 'B'
        accuracyArray(1,r1) = accuracy;
        r1 = r1 + 1;
    elseif imagePath(ind) == 'H'
        accuracyArray(2,r2) = accuracy;
        r2 = r2 + 1;
    elseif imagePath(ind) == 'F'
        accuracyArray(3,r3) = accuracy;
        r3 = r3 + 1;
    elseif imagePath(ind) == 'M'
        accuracyArray(4,r4) = accuracy;
        r4 = r4 + 1;
    end
end

% Calculate overall accuracy and average reaction time
% overallAccuracy = sum(~isnan(accuracyArray),2)./(numTrials/4);
overallAccuracy = sum(accuracyArray,2)./(numTrials/4);
averageReactionTime = nanmean(reactionTimeArray);

% Store overall accuracy and average reaction time
subjectAccuracy(1:4,block_number) = overallAccuracy;
subjectAccuracy(5,block_number) = averageReactionTime;

% Save updated data
save('C:\Users\USER\Desktop\darsi\courses\FoNSc\pnas_package\DS\human\results\subject1\subjectAccuracy.mat','subjectAccuracy');

% Close Psychtoolbox window
sca;



useSVM = 1;
if useSVM
%   Model = CLSoAsusvm(XTrain,ytrain);  %training
  Model = CLSosusvm(XTrain,ytrain);
  [ry,rw] = CLSosusvmC(XTest,Model); %predicting new labels
else %use a Nearest Neighbor classifier
  Model = CLSnn(XTrain, ytrain); %training
  [ry,rw] = CLSnnC(XTest,Model); %predicting new labels
end  
successrate = mean(ytest==ry) %a simple classification score


