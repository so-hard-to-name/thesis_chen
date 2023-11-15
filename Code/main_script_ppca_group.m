% Load the CSV file
data = readtable('filtered_data1.csv'); % Replace 'your_data.csv' with the actual file path

% Get unique IDs to create groups
uniqueIDs = unique(data.stay_id);

% Initialize a new table for imputed data
imputedData = data;

% Loop through each unique ID group
for i = 1:length(uniqueIDs)
    disp(uniqueIDs(i));
    % Extract the current group based on ID
    currentGroup = data(data.stay_id == uniqueIDs(i), :);
    
    % Sort the current group by the 'Series' column
    currentGroup = sortrows(currentGroup, 'hour_num');
    
    % Extract the data columns for imputation
    dataColumns = currentGroup{:, 6:27};
    array = dataColumns;
    [C, ss, M, X,Ye] = ppca_mv(array,2,0);
    imputedDataColumns = array2table(Ye);
    disp(imputedDataColumns);
    disp('line break');
    disp(imputedData(imputedData.stay_id == uniqueIDs(i), 6:27))
    % Replace the imputed data back into the original table
    imputedData(imputedData.stay_id == uniqueIDs(i), 6:27) = imputedDataColumns;
end

% Write the imputed data to a new CSV file
writetable(imputedData, 'imputed_ppca_group.csv'); % Replace 'imputed_data.csv' with the desired output file name

disp('Imputation completed and saved to imputed_data.csv');
