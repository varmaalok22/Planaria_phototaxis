%% Circular Arena Cropping Stuff % for fine categories, modify last line
clear all; close all; clc;

%% Directory etc. - Load data (already pooled)
directory = 'Z:\120mm\Training shiwangi\Automated\'; % Where the pooled data is stored.
data_filename = '2smallworms_Pooled.xlsx';
pooled_data = readtable(fullfile(directory, data_filename));
worm_ids = unique(pooled_data.Worm_ID);
no_of_worms = length(worm_ids);

bg_folders = dir(directory);
bg_folders = bg_folders([bg_folders.isdir]);
bg_folders = bg_folders(3:end);
no_of_bgs = length(bg_folders);

assert(no_of_worms == no_of_bgs, 'The no of worms and no of bg images are unequal! Check folder and data.');

%% Set up important variables.
% If you want to change the arena, and find a new limit for the radius of
% the arena, uncomment the following code.
test_bg_path = 'Z:\120mm\Training shiwangi\Automated\Frames01\bg.tif'; % Put complete path of a test bg image here.
test_bg = imread(test_bg_path);
imshow(test_bg); d = 743; % Manually find a value for d, and fix it here.
% 
radius_range = [370 380]; % Change this using reference values of d (if needed).
% 
% Trying to find an appropriate sensitivity for this set of images.
% 
circleFound = false;
sensitivities_to_test = linspace(0.8, 0.99, 100);
t = 1;
while ~circleFound
    sensitivity = sensitivities_to_test(t);
    [C, R] = imfindcircles(test_bg, radius_range, 'ObjectPolarity', ...,
        'dark', 'Sensitivity', sensitivity);
    [C, R] = imfindcircles(test_bg, radius_range, 'ObjectPolarity', ...,
        'bright', 'Sensitivity', sensitivity);
    if ~isempty(C)
        circleFound = true;
        break
    elseif t<length(sensitivities_to_test)
        t = t+1;
    else
        break
    end
end
fprintf('The minimum sensitivity to use is %.4f \n', sensitivity);
imshow(test_bg); viscircles(C,R); % To check if things detected properly.

%%
for i=1:no_of_worms
%% Segment out circular arena for this specific worm (using bg_worm#)
curr_dir = bg_folders(i).name;
curr_bg = imread(fullfile(directory, curr_dir, 'bg.tif'));
curr_worm_id = worm_ids(i);

[centres, radii] = imfindcircles(curr_bg, radius_range, 'ObjectPolarity', ...,
    'dark', 'Sensitivity', 0.997);
% [centres, radii] = imfindcircles(curr_bg, radius_range, 'ObjectPolarity', ...,
%     'bright', 'Sensitivity', 0.993);
figure;
imshow(curr_bg); viscircles(centres, radii, 'Color', 'r'); % Red is the arena.
factor = 0.85; % Percentage of the radius that is a threshold.
threshold = radii*factor;
hold on; viscircles(centres, threshold, 'Color', 'b');
bg_boundaries = getframe;
imwrite(bg_boundaries.cdata, fullfile(directory, strcat(curr_worm_id{:},'_boundaries.tif')));
close all;

%% Get relevant information
curr_worm_data = pooled_data(strcmp(curr_worm_id{:},pooled_data.Worm_ID), :);
centroid_x = curr_worm_data.Centroid_x;
centroid_y = curr_worm_data.Centroid_y;
curr_worm_data.Centre_Distance = sqrt((centroid_x-centres(1)).^2 + (centroid_y-centres(2)).^2);
% filtered_curr_worm_data = curr_worm_data(curr_worm_data.Centre_Distance <= threshold, :);

nframes = height(curr_worm_data);
filtered_idxs = nan(1, nframes);
for j=1:nframes
    if curr_worm_data.Centre_Distance(j) <= threshold
        filtered_idxs(j) = 1;
    else
        break
    end
end
filtered_curr_worm_data = curr_worm_data(~isnan(filtered_idxs),:);

if ~exist('filtered_pooled_data', 'var')
    filtered_pooled_data = filtered_curr_worm_data;
else
    filtered_pooled_data = [filtered_pooled_data; filtered_curr_worm_data];
end
% imshow(curr_bg); hold on;
% viscircles(centres, threshold, 'Color', 'b');
% plot(filtered_curr_worm_data.Centroid_x, filtered_curr_worm_data.Centroid_y)
end

writetable(filtered_pooled_data, fullfile(directory, 'Filtered_Pooled_Data.xlsx'));% change to Filtered_Pooled_Data_finecats.xlsx for fine categories