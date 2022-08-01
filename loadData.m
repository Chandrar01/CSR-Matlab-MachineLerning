function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   http://networkrepository.com/pima-indians-diabetes.php
%   and splits the dataset into two sets: training set and testing set
%

 % ====================== YOUR CODE HERE ======================
    % Instructions: Import spreadsheets data, extract the first
    % 8 columns and store them as X. Extract the last column and 
    % store it as y. 
    %
    % Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    %
    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    %

     filename = 'Heart-Failure-Prediction-Data.csv';
   
    T = readtable(filename);
    total_no_rows = height(T);
    trainingRows = round(total_no_rows*0.7);
   
    % making a Set of all row numbers
    
    All_rows = 1:1:total_no_rows;
    
    % making a set of randomly selected rows of training set
    set_of_training_rows = randperm(total_no_rows,trainingRows);
    
    TrainingSet = T(set_of_training_rows,:); 
    
   % Remaining rows are the testing set 
    set_of_testing_rows = setdiff(All_rows,set_of_training_rows);
    
    TestingSet = T(set_of_testing_rows,:);
    
    X_train = table2array(TrainingSet(:,1:12));
  
    y_train = table2array(TrainingSet(:,13));
   
    X_test = table2array(TestingSet(:,1:12));
   
    y_test = table2array(TestingSet(:,13));
    
    


% ============================================================
end