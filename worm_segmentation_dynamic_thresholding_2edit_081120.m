%% Worm Segmentation and Tracking Code

% Aalok Varma
% April 06, 2019.
% modified for reducing uniform intensity on 30-10-2020
%sometimes can lead to swapping of head and tail  THE RANGE OF SWAPPING
%NEEDS TO BE NOTED AND CORRECTED USING 'Reassign_incorrect_head_tail.m' code

%%%% This is the core video analysis file. It segments the worm, identifies
%%%% head and tail semi-manually (first frame only). It gives information
%%%% about the centroid, head and tail coordinates, and two key angles -
%%%% orientation and head turn angle. 
%%%% 
%%%% Head vector is the vector from the centroid to the head.
%%%% Tail vector is the vector from the tail to the centroid.

%%%% Orientation angle is the angle of the tail vector.
%%%% Head turn angle is the angle between the head and tail vectors.

%%% Note about orientation vector.
%%% Orientation vector is from tail to centroid.
%%% Excel sheet stores *this* as orientation, and NOT what MATLAB gives,
%%% which is useless anyway.
tic
clearvars; close all; clc;

%% Set up directories etc.
dataDir = 'C:\Users\LABUSER\Desktop\caustics\Automated\'; % Make sure to end this with a backslash.
outDir=strcat(dataDir,'Processed\'); % Make sure to end this with a backslash.
% outDir=strcat(dataDir,'Processed_no_reassignment\'); % Make sure to end this with a backslash.
if ~exist(outDir)
    mkdir(outDir)
end


file_basename = 'test';

% Give a unique ID to the worm you are analysing.
% Do NOT remove the curly braces.
worm_ID ={'WT_01'};

dataOut_filename = strcat(outDir, worm_ID{1}, '.xlsx');
filt_dataOut_filename = strcat(outDir, worm_ID{1}, '_filtered.xlsx');
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

% Paramter for dynamic thrsholding. This sets how many standard deviations
% away from the mean of the peak in the image histogram would be
% consideredn
% object and not background. Change this to improve segmentation, if needed.
n_std = 7;

numDivisions = 3; % Do not change.
startFrame = 1;
endFrame = numFiles;
n = endFrame-startFrame+1;

%% Find limits of the circular arena
radius_range = [315 335]; % Change this using reference values of d (if needed).

sensitivities = 0.99; % CHANGE THIS VALUE IF NO CIRCLE WAS FOUND.
for i=1:length(sensitivities)
    s = sensitivities(i);
    [centres, radii] = imfindcircles(backgroundImage, radius_range, 'ObjectPolarity', ...,
        'dark', 'Sensitivity', s, 'EdgeThreshold', 0.1);
    %fprintf('%d\n', length(radii));
end

assert(length(radii) == 1, 'Fewer or more than one circle was found.');

figure;
imshow(backgroundImage); viscircles(centres, radii, 'Color', 'r'); % Red is the arena.
factor = 0.85; % Percentage of the radius that is a threshold.
threshold = radii*factor;
hold on; viscircles(centres, threshold, 'Color', 'b');
bg_boundaries = getframe;
% imwrite(bg_boundaries.cdata, fullfile(directory, strcat(curr_worm_id{:},'_boundaries.tif')));
close all;

%% Make a mask from the circle coordinates. Credit to Jonas for original code.
x1 = centres(1); y1 = centres(2);

[rNum,cNum,~] = size(backgroundImage);

[xx,yy] = ndgrid((1:rNum)-y1,(1:cNum)-x1);
mask = (xx.^2 + yy.^2)<radii^2;

% imshowpair(backgroundImage, mask, 'montage');
% img = backgroundImage;
% img(mask) = uint8(1);
% imshow(img)

%% Initialize storage variables
all_centroids = nan(n, 2);
all_heads = nan(n, 2);
all_tails = nan(n, 2);
all_headAngles = nan(n, 1);
all_orientations = nan(n, 1);
contourLengths = nan(n, 1);

%% Process all frames
numProcessed = 0;
n_prev = 3;

prevHead = [nan nan];
prevTail = [nan nan];
prevCentroid = [nan nan];

prevHeads = nan(n_prev, 2);
prevTails = nan(n_prev, 2);
prevCentroids = nan(n_prev, 2);

f_thr_crossed = nan;

for i=startFrame:endFrame
    filename = filesArray(i).name;
    output_filename = strcat(filename(1:end-4),'_track.tif');
    currFrame = imread(filename);
    try
        currFrame = rgb2gray(currFrame);
    end
    
    
    % Old code.
    % Background subtraction.
    % frame = imsubtract(backgroundImage, currFrame);
    % Thresholding the image
    % BW = imbinarize(frame, threshold/255);
    
    % New code. Written on October 21st, 2020.
    % The image histograms for all these videos have 2 major peaks.
    % To segment the worm, use the image histograms to find the peaks
    % intensities of bright and dark. The worm has intensities intermediate
    % to the two peaks.
    
    % Mask the current frame with the circular limits of the arena and then
    % find the threshold.
    currFrame(~mask) = 0;
    
    peak1 = fitdist(currFrame(currFrame>0), 'normal');
    thr1 = peak1.mu - n_std * peak1.sigma;
    %peak2 = fitdist(currFrame(currFrame<thr1), 'normal');
    % imshow(currFrame <= x2.mu + 2*x2.sigma)
    %thr2 = peak2.mu + 2*peak2.sigma;
    thr2 = 0;
    
    BW = (currFrame > thr2 & currFrame <= thr1);
    
    % Morphological operations to clean up the image and extract just the
    % worm.
    BW=bwareaopen(BW, 30); % Minimum area of objects should be 15 pixels.
    BW=imfill(BW,'holes'); % Fill holes in objects, if any.
    [L,numobj] = bwlabel(BW);
    stats = regionprops(L, currFrame, 'Area', ...,
        'PixelList', 'Centroid');
    if ~isempty([stats.Area])
        % For the first frame, find the centroid closest to the centre. For
        % the later frames, find the centroid closest to the previous
        % centroid.
        centroids = reshape([stats.Centroid], 2, [])';
        if isnan(prevCentroid)
            [~, idx] = min(pdist2(centres, centroids));
        else
            [~, idx] = min(pdist2(prevCentroid, centroids));
        end
        Centroid = stats(idx).Centroid;
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
    
%     if ~isnan(prevHead)
%         hhDist = norm(wormHead - prevHead);
%         ttDist = norm(tail - prevTail);
%         htDist = norm(wormHead - prevTail);
%         thDist = norm(tail - prevHead);
%         
%         % Head and tail have been misidentified if
%         
%         if (hhDist > htDist) && (ttDist > thDist)
%             dummy = tail;
%             tail = wormHead;
%             wormHead = dummy;
%         end
%     end

    if ~isnan(prevHead)
        hhDist = sum(pdist2(wormHead, prevHeads));
        ttDist = sum(pdist2(tail, prevTails));
        htDist = sum(pdist2(wormHead, prevTails));
        thDist = sum(pdist2(tail, prevHeads));
        
        % Head and tail have been misidentified if
        
        if (hhDist > htDist) && (ttDist > thDist)
            dummy = tail;
            tail = wormHead;
            wormHead = dummy;
        end
    end
    
    if isnan(prevHeads)
        prevHeads = repmat(wormHead, n_prev, 1);
        prevTails = repmat(tail, n_prev, 1);
        prevCentroids = repmat(Centroid, n_prev, 1);
    else
        prevHeads(1:end-1,:) = prevHeads(2:end,:);
        prevTails(1:end-1,:) = prevTails(2:end,:);
        prevCentroids(1:end-1,:) = prevCentroids(2:end,:);
        
        prevHeads(end,:) = wormHead;
        prevTails(end,:) = tail;
        prevCentroids(end,:) = Centroid;
    end
    
    all_heads(i, :) = wormHead;
    all_tails(i, :) = tail;
    prevHead = wormHead;
    prevTail = tail;
    prevCentroid = Centroid;
    
    % If the centroid has crossed the threshold, then note the frame
    if pdist2(Centroid, centres) > threshold
        f_thr_crossed = numProcessed + 1;
    end
    
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
% ylim([0 rNum]); xlim([0 cNum]);
colormap('jet')
colorbar
title('Orientation Angle with respect to arena');

figure
scatter(all_centroids(:,1), all_centroids(:,2), 12, all_headAngles, 'filled');
set(gca, 'ydir', 'reverse')
% ylim([0 rNum]); xlim([0 cNum]);
colormap('jet')
colorbar
title('Head Turn Angle with respect to arena');

%% Saving all the data
Worm_ID = cell(n, 1);
Worm_ID(:) = worm_ID;
Frame_No = (startFrame:endFrame)';
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

if ~isnan(f_thr_crossed)
    filtered_data = all_data(1:f_thr_crossed, :);
    writetable(filtered_data, filt_dataOut_filename);
end

toc