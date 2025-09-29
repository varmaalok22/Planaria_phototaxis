%% Pooled Polar Histogram
% Aalok Varma
% February 24, 2019.

clearvars; close all; clc;

%% Set directory shizz
main_directory = 'Z:\120mm\Training shiwangi\Automated\';
output_filename = '2smallworms'; %Input shortform of the name of experiment here.

frame_rate = 6.6; % Input the frame rate 
scale_factor = 6.19; % 6.64 for the 120mm assay fpr this particular experiment 8.1467 for rectangulatr pixels/mm
time_interval = 1/frame_rate;

output_filename = strcat(output_filename, '_Pooled.xlsx');
%main_directory = 'E:\Rimple\Automated tracking\left half ovo 07022019\505ldarkr auto tracking\ovo\';
genotype = strsplit(main_directory, '\');
genotype = mat2str(cell2mat(genotype(end-1)));
data_folders = dir(main_directory);
data_folders = data_folders([data_folders.isdir] == 1);

clear('pooled_data');
s = length(data_folders);

%% Compile data from directories - calculate distances and assign categories.
for i=3:s
    current_directory = strcat(main_directory, data_folders(i).name, '\Processed\');
    filename = dir(fullfile(current_directory, '*.xlsx'));
    filename = filename.name;
    filepath = fullfile(current_directory, filename);
    new_data = readtable(filepath);
    [rows, cols] = size(new_data);
    new_data.ScaleFactor = ones(rows, 1)*scale_factor;
    
    % Calculate distances and velocities.
    x1 = new_data.Centroid_x(1:end-1);
    y1 = new_data.Centroid_y(1:end-1);
    x2 = new_data.Centroid_x(2:end);
    y2 = new_data.Centroid_y(2:end);
    dist = sqrt((x2-x1).^2 + (y2-y1).^2);
    dist = [0; dist];
    new_data.Distance = dist;

    vel = dist./time_interval; % This is velocity in pixels per second.
    new_data.Velocity = vel;
    
    % Assign Orientation Category to each frame.
    orientation_groups = [180 0 -90 90 45 135]; % Set this appropriately
    errors = [30 30 30 30 15 15];

    Orient = nan(rows, 1);
    for r=1:rows
        orientation = new_data.Orientation(r);
        if abs(orientation) >= 150 && abs(orientation) <= 180
            Orient(r) = 'A';
        elseif abs(orientation) >= 0 && abs(orientation) < 30
            Orient(r) = 'B';
        elseif orientation <= -60 && orientation > -120
            Orient(r) = 'C';
        elseif orientation >= 60 && orientation < 120
            Orient(r) = 'D';
        elseif abs(orientation) >= 30 && abs(orientation) < 60
            Orient(r) = 'E';
        elseif abs(orientation) >= 120 && abs(orientation) < 150
            Orient(r) = 'F';
        end
    end

    Orient = char(Orient);
    new_data.Category = categorical(cellstr(Orient));
    
    if ~exist('pooled_data', 'var')
        pooled_data = new_data;
    else
        pooled_data = [pooled_data; new_data];
    end
end

%% Calculate distances and velocities here.

writetable(pooled_data, fullfile(main_directory, output_filename));

%% Pooled Polar Histogram - Finally!!!
% figure;
% hold on
histogramHeight = 0.2;
p = polarhistogram(deg2rad(pooled_data.Orientation), 40, 'Normalization', 'probability', 'DisplayName', genotype);

% To change the axis limits
rlim([0 histogramHeight])

% p = polarhistogram(ANGLES, 40, 'Normalization', 'pdf', 'DisplayName', genotype)
title("Pooled orientation histogram");
legend;

% If you want to verify that all probabilities sum up to 1, uncomment this:
% sum(p.Values)

hold on;