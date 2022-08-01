function [beta1, beta2, J_history] = trainNN(X, y, beta1, beta2, alpha, num_epochs)
%TRAINNN train the neural network model using backpropagation algorithm. It
%updates the weights, beta1 and beta2 using the training examples. It also
%generates the cost computed after each epoch. 

% useful values
[n, ~] = size(X); % n is number of training examples
num_hidden = length(beta1(:,1)); % number of hidden units (bias not included)
num_output = length(beta2(:,1)); % number of output units

J_history = zeros(num_epochs,1); % 

% Add intercept term to X_train
X = [ones(n, 1) X];  % 538x9

for epoch = 1:num_epochs
% for each training example, do the following
    Jd = 0;
    for d = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the weights beta1 and
    %               beta2. The key steps are indicated as follows
    %
    %
   
 
        %% Step 1: forward propagate to generate the network output
        
        %for layer 1 input
        x = X(d,:)';
        
        %for layer 2
        
        a2 = sigmoid(beta1*x); % added bias
        
        a3 = sigmoid(beta2* [1;a2]);
       
        
        %% Step 2: for each output unit, calculate its error term
        % Recall that the number of output units is num_output

        delta3 = a3 * (1 - a3)*(y(d) - a3);
        
        
        %% Step 3: for each hidden unit, calculate its error term
        % Recall that number of hidden units is num_hidden+1
        
        for i = 1:num_hidden
        delta2(i,1) = a2(i,1) * (1 - a2(i,1))* delta3.* beta2(1,i);
        end
        

        %% Step 4: update the weights using the error terms
        
        % update weights of layer 1
        
        for i = 1:13   
            for j = 1:12
                beta1(j,i) = beta1(j,i) + alpha * (delta2(j) * x(i));
            end   
        end
        
        % update weight of layer 2
        a2 = [1;a2];
        for i = 1:12  
            beta2(1,i) = beta2(1,i) + alpha * (delta3 * a2(i));   
        end
        
        
        %% calculate the cost per epoch
        Jd = Jd + (y(d) - a3)^2;
    end
    J_history(epoch) = Jd/2;
end