%% SDSU Machine Learning Course (EE600/CompE596)
%% Term Project Programming :  Neural Network 
%
%  Instructions
%  ------------
% 
%  Dataset comes from: 
%   https://www.kaggle.com/fedesoriano/heart-failure-prediction
%
%  You will need to complete the following functions in this 
%  assignment
%
%     loadData.m
%     featureNormalize.m
%     trainNN.m
%     evaluateAccuracy.m
%     predict.m
%     sigmoid.m
%
%  For this part of the Term project, we are reusing the Code we did as a 
%  part of Neural natwork Assignment 


% Initialization
clear ; close all; clc

%% =========== Part 1: Data Preprocessing =============
% Instructions: The following code loads data into matlab, splits the 
%               data into two sets, and performs feature normalization. 
%               You will need to complete code in loadData.m, and 
%               featureNormalize.m
%

%% Load data
fprintf('Loading data ...\n');

% ====================== YOUR CODE HERE ======================
[X_train, y_train, X_test, y_test] = loadData();

% ============================================================

[n, m] = size(X_train); % n is the number of total data examples
                        % m is the number of features
                        
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');


%% Normalize the features. 
% ====================== YOUR CODE HERE ======================
[Xn_train, mu, sigma] = featureNormalize(X_train);

% ============================================================

num_train = length(y_train); % number of training examples

%% ================ Part 2: Training Neural Network ============

fprintf('Running backpropagation ...\n');

% Instructions: The following code applies backpropagation algorithm to 
%               estimate the parameters in a neural network model with 
%               a single hidden layer. 
%               You should complete code in trainNN.m, and sigmoid.m
%
%               Try running the backpropagation algorithm with 
%               different values of alpha and num_hidden and see which 
%               one gives you the best result.
%

%% Setup the parameters you will use for this assignment
% ====================== YOUR CODE HERE ======================
alpha = 0.001; % you should change this
num_epochs = 10000; % you should change this 

num_hidden = 12;  % number of hidden units (not including the bia unit)
num_output = 1;  % number output unit

% ============================================================

% initialize the weights: beta1 and beta2 
beta1 = rand(num_hidden, m+1); % weights associated with links between input and hidden layers
beta2 = rand(num_output, num_hidden+1); % weights associated with links between hidden and output layers

%% Run backpropagation 
% ====================== YOUR CODE HERE ======================
% Add intercept term to X_train and X_test

[beta1, beta2, J_history] = trainNN(Xn_train, y_train, beta1, beta2, alpha, num_epochs);

% ============================================================

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of epochs');
ylabel('Cost J');
%% ========== Part 3: Evaluate performance =============

fprintf('Evaluate the prediction accuracy ...\n');

% Instructions: The following code evaluates the performance of
%               the trained neural network model. You should 
%               complete code in evaluateAccuracy.m, predict.m
%

num_test = length(y_test); % number of testing examples

% normalize input features of the testing set
Xn_test = (X_test - mu)./sigma;

% ====================== YOUR CODE HERE ======================
accuracy = evaluateAccuracy(beta1, beta2, Xn_test, y_test);

% ============================================================

% Display the prediction accuracy
fprintf('Accuracy:\n %f\n', accuracy);
fprintf('\n');


% ==============================================================
%   This is additional tst data which is used to understand if NN 
%   model predict accuratly result based on smoking habit feature
%   as this is one of the definitive features we have considered
% ==============================================================

T1 =[54	1	2	150	195	1	1	122	1	0	1	0;
     39	1	2	120	339	1	1	170	1	0	1	0;
     54	1	1	110	208	1	1	142	1	0	1	0;
     37	2	2	130	211	1	1	142	1	0	1	0;
     42	2	2	115	211	1	2	137	1	0	1	0;
     54	2	1	120	273	1	1	150	1	1.5	2	0;
     43	2	1	120	201	1	1	165	1	0	1	0;
     43	2	3	100	223	1	1	142	1	0	1	0;
     44	1	1	120	184	1	1	142	1	1	2	0;
     49	2	1	124	201	1	1	164	1	0	1	0;
     40	1	2	130	215	1	1	138	1	0	1	0;
     36	1	2	130	209	1	1	178	1	0	1	0;
     53	1	4	124	260	1	2	112	2	3	2	0;
     56	1	2	130	167	1	1	114	1	0	1	0;
     32	1	1	125	254	1	1	155	1	0	1	0;
     41	2	1	110	250	1	2	142	1	0	1	0;
     48	2	4	150	227	1	1	130	2	1	2	0;
     54	2	1	150	230	1	1	130	1	0	1	0];

p1 = predict(beta1, beta2, T1);

display(p1)

% data with smoking habit 1

T2 = [25	1	3	100	146	1	1	120	1	0	1	1];

p2 = predict(beta1, beta2, T2);

fprintf('prediction : %d\n', p2);

% same data with smoking habit 0

T3 = [25	1	3	100	146	1	1	120	1	0	1	0];

p3 = predict(beta1, beta2, T3);

fprintf('prediction : %d\n', p3);


