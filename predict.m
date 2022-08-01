function p = predict(beta1, beta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(beta1, beta2, X) outputs the probability of the output to 
%   1, given input X and trained weights of a neural network (beta1, beta2)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. 
%
%
    X = [ones(size(X, 1), 1) X]; 

    a_1 = X; 
    z_2 = a_1 * beta1'; 
    a_2 = sigmoid(z_2); 
    a_2 = [ones(size(a_2, 1),1) a_2]; 
    z_3 = a_2 * beta2'; 
    g = sigmoid(z_3);

    for i = 1:m
        if g(i,1) >= 0.5
            p(i,1) = 1;
        else
            p(i,1) = 0;
        end   
    end


% =========================================================================


end
