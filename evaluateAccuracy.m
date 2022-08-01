function accuracy = evaluateAccuracy(beta1, beta2, X, y)
%EVALUATEACCURACY calculates the prediction accuracy of the learned 
%neural network model using the testing data 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the percentage of accurately predicted examples 
%
%
num = size(X, 1);
pred = predict(beta1,beta2, X);
acc = zeros(num , 1);
   

for i = 1:num
    
    if y(i,1)==pred(i,1)
        acc(i,1) = 1;
    else
        acc(i,1) = 0;
    end    
end

accuracy = mean(acc)*100;

% ============================================================

end