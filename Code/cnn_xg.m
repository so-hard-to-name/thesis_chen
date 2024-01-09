% 1. Read the CSV file
data = readtable('imputed_li.csv');

% Find unique 'stay_id's
uniqueIDs = unique(data.stay_id);

% Initialize a cell array to store groups
groups = cell(length(uniqueIDs), 1);

% Split the data into groups based on 'stay_id'
for i = 1:length(uniqueIDs)
    currentID = uniqueIDs(i);
    groups{i} = data(data.stay_id == currentID, :);
end

% 4. Prepare data for CNN
cnn_data = cellfun(@(x) table2array(x(:, 7:end)), groups, 'UniformOutput', false);

% Now cnn_data contains the data for your CNN model, structured as a cell array where each cell holds the data for one group
% Assuming you've already processed the data as in the previous code snippet

% To print the first group
disp(cnn_data{1}); % This will print the first group in the CNN data
firstGroupID = groups{1}.stay_id(1);
disp(firstGroupID);
