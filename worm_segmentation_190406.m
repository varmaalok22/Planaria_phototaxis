%% Worm Segmentation and Tracking Code

% Aalok Varma
% April 06, 2019.
% 

%%%% This is the core video analysis file. It segments the worm, identifies
%%%% head and tail semi-manually (first frame only). It gives information
%%%% about the centroid, head and tail coordinates, and two key angles -
%%%% orientation and head turn angle. 
%%%% 
%%%% Head vector is the vector from the centroid to the head.
%%%% Tail vector is the vector from the tail to the centroid.

%%%% Orientation angle is the angle of the tail vector.
%%%% Head turn angle is the angle between the head and tail vectors.

%%% Note about orientation vector.n
%%% Orientation vector is from tail to centroid.
%%% Excel sheet stores *this* as orientation, and NOT what MATLAB gives,
%%% which is useless anyway.
tic
clearvars; close all; clc;

%% Set up directories etc.
dataDir = 'Z:\120mm\Training shiwangi\Automated\Frames02\'; % Make sure to end this with a backslash.
outDir=strcat(dataDir,'Processed\'); % Make sure to end this with a backslash.
if ~exist('outDir')
    mkdir(outDir)
end

file_basename = 'test';

% Give a unique ID to the worm you are analysing.
% Do NOT remove the curly braces.
worm_ID ={'WT_02'};

dataOut_filename = strcat(outDir, worm_ID{1}, '.xlsx');
extension = '.tif';
filesArray=dir(strcat(dataDir,file_basename,'*',extension));
numFiles=length(filesArray);
cd(dataDir)

%% Set up necessary parameters for segmentation and processing
backgroundImage = imread('bg.tif'); 
try
    backgroundImage = rgb2gray(backgroundImage);
end
saveImages = 1; % 0 or 1, to save images post-processing.
saveEvery = 20; % Save every nth frame

% To change, and optimize per worm.
threshold = 20;

threshold_high = threshold + 30; % Play around with this if error in line 231.
numDivisions = 3; % Do not change.
startFrame = 1;
stepSize = 1;
endFrame = numFiles;
n = endFrame-startFrame+1;

%% Initialize storage variables
all_centroids = nan(n, 2);
all_heads = nan(n, 2);
all_tails = nan(n, 2);
all_headAngles = nan(n, 1);
all_orientations = nan(n, 1);
contourLengths = nan(n, 1);

%% Process all frames
numProcessed = 0;
prevHead = [nan nan];
prevTail = [nan nan];

for i=startFrame:stepSize:endFrame
    filename = filesArray(i).name;
    output_filename = strcat(filename(1:end-4),'_track.tif');
    currFrame = imread(filename);
    try
        currFrame = rgb2gray(currFrame);
    end
    
    % Background subtraction.
    frame = imsubtract(backgroundImage, currFrame);
    
    % Thresholding the image
    BW = imbinarize(frame, threshold/255);
    
    % Morphological operations to clean up the image and extract just the
    % worm.
    BW=bwareaopen(BW, 15); % Minimum area of objects should be 15 pixels.
    BW=imfill(BW,'holes'); % Fill holes in objects, if any.
    [L,numobj] = bwlabel(BW);
    stats = regionprops(L, currFrame, 'Area', ...,
        'PixelList', 'WeightedCentroid');
    if ~isempty([stats.Area])
        % Find the maximum area, which is the object.
        [~, idx] = max([stats.Area]);
        Centroid = stats(idx).WeightedCentroid;
        all_centroids(i,:) = Centroid;
        pixels = stats(idx).PixelList;
        worm_pixels = fliplr(pixels(1,:));
        worm_label = L(worm_pixels(1), worm_pixels(2));
    end
    L = (L == worm_label); % This is the segmented worm.
    
    % Get the perimeter of the worm.
    wormOutline = bwperim(L);
    perimPixels = find(wormOutline);
    [py, px] = ind2sub(size(L), perimPixels);
    
    % Ordered contour of the worm's outline.
    orderedContour = bwtraceboundary(wormOutline, [py(1) px(1)], 'N');
    pixelDist = 3; % No of pixels between which to calculate angles.
    
    % Calculate the angles between the pixels.
    s = length(orderedContour);
    contourLengths(i,:) = s;
    curvatures = nan(1, s);
    
    for j1=1:s
        j0 = j1 - pixelDist;
        j2 = j1 + pixelDist;
        
        p1 = orderedContour(j1,:);
        
        if j0>=1
            p0 = orderedContour(j0, :);
        else
            p0 = orderedContour(j0+s, :);
        end
        
        if j2 <= s
            p2 = orderedContour(j2, :);
        else
            p2 = orderedContour(j2-s, :);
        end
        
        v10 = p0 - p1; % Vector from p1 to p0.
        v12 = p2 - p1; % Vector from p1 to p2.
        
        % Angle between the 2 vectors.
        curvature = atan2(v10(2), v10(1)) - atan2(v12(2), v12(1));
        curvatures(j1) = curvature;
    end
    curvatures = unwrap(curvatures); % Converts from [-pi, pi] to [0, 2pi].
    curvatures = curvatures - mean(curvatures);
    
    % Find the point(s) of maximum angle. If it is the first frame, then
    % get the user to approve the segmentation of head vs. tail. If it's
    % not the first frame, then check whether it is closer to the head or
    % the tail of the previous frame. Assign the point as head or tail
    % accordingly.
    headIndex = find(curvatures == max(curvatures));
    if length(headIndex) > 1
        headIndex = headIndex(1);
    end
    
    wormHead = orderedContour(headIndex, :);
    
    curvaturesReduced = curvatures;
    
    index0 = headIndex - ceil(s/numDivisions);
    index1 = headIndex + ceil(s/numDivisions);
    if index1 > s
        index1 = index1 - s;
        curvaturesReduced(headIndex:s) = 0;
        curvaturesReduced(1:index1) = 0;
    else
        curvaturesReduced(headIndex:index1) = 0;
    end
    if index0 < 1
        index0 = index0 + s;
        curvaturesReduced(index0:s) = 0;
        curvaturesReduced(1:headIndex) = 0;
    else
        curvaturesReduced(index0:headIndex) = 0;
    end
    
    tailIndex = find(curvaturesReduced == max(curvaturesReduced));
    if length(tailIndex) > 1
        tailIndex = tailIndex(1);
    end
    
    tail = orderedContour(tailIndex, :);
    
    if numProcessed == 0 % Get user to confirm whether head or tail.
        figure;
        imshow(currFrame); hold on;
        plot(wormHead(2), wormHead(1), 'b*');
        plot(tail(2), tail(1), 'g*');
        plot(Centroid(1), Centroid(2), 'r*');
        answer = input('Is this the correct head/tail segmentation? Y/N:', 's');
        if strcmpi(answer, 'N')
            % Reassign head and tail.
            dummy = tail;
            tail = wormHead;
            wormHead = dummy;
        end
        close;
    end
    
    if ~isnan(prevHead)
        hhDist = norm(wormHead - prevHead);
        ttDist = norm(tail - prevTail);
        htDist = norm(wormHead - prevTail);
        thDist = norm(tail - prevHead);
        
        % Head and tail have been misidentified if
        
        if (hhDist > htDist) && (ttDist > thDist)
            dummy = tail;
            tail = wormHead;
            wormHead = dummy;
        end
    end
    
    all_heads(i, :) = wormHead;
    all_tails(i, :) = tail;
    prevHead = wormHead;
    prevTail = tail;
    
    % Save the image of the segmentation
    if saveImages
        if mod(i, saveEvery) == 0
            figure;
            imshow(currFrame); hold on;
            plot(wormHead(2), wormHead(1), 'b*');
            plot(tail(2), tail(1), 'g*');
            plot(Centroid(1), Centroid(2), 'r*');
            saveas(gcf, strcat(outDir, output_filename));
            close;
        end
    end

%% Calculate heading and orientation with respect to TAIL-CENTROID vector
    tailVector = Centroid - fliplr(tail);
    headVector = fliplr(wormHead) - Centroid; % This is the heading direction.
    
    orientation = atan2d(tailVector(2), tailVector(1));
    headAngle = atan2d(tailVector(2)*headVector(1)-headVector(2)*tailVector(1), ...,
        tailVector(1)*headVector(1)+tailVector(2)*headVector(2));
    
    all_headAngles(i,:) = headAngle;
    all_orientations(i,:) = orientation;

numProcessed = numProcessed + 1;
end

%% Plot Orientation and Head Angles with respect to arena
figure
scatter(all_centroids(:,1), all_centroids(:,2), 12, all_orientations, 'filled');
set(gca, 'ydir', 'reverse')
colormap('jet')
colorbar
title('Orientation Angle with respect to arena');

figure
scatter(all_centroids(:,1), all_centroids(:,2), 12, all_headAngles, 'filled');
set(gca, 'ydir', 'reverse')
colormap('jet')
colorbar
title('Head Turn Angle with respect to arena');

%% Saving all the data
Worm_ID = cell(n, 1);
Worm_ID(:) = worm_ID;
Frame_No = (startFrame:stepSize:endFrame)';
Centroid_x = all_centroids(:,1);
Centroid_y = all_centroids(:,2);
Head_x = all_heads(:,2); % Head and Tail coordinates are reversed. MATLAB weirdness. This is not a mistake.
Head_y = all_heads(:,1);
Tail_x = all_tails(:,2);
Tail_y = all_tails(:,1);
Head_Angle = all_headAngles;
Orientation = all_orientations;

all_data = table(Worm_ID, Frame_No, Centroid_x, Centroid_y, Head_x, ..., 
    Head_y, Tail_x, Tail_y, Head_Angle, Orientation);
writetable(all_data, dataOut_filename);
toc
