dataall12hvitalsigndemo = readtable('data_all12h_vitalsign_demo.csv');
columns = 6:27;

data = dataall12hvitalsigndemo(:, columns);
array = table2array(data);

[C, ss, M, X,Ye] = ppca_mv(array,2,1);

dataall12hvitalsigndemo(:, columns) = array2table(Ye);

writetable(dataall12hvitalsigndemo, 'imputed_ppca.csv');