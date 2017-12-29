function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples
n = length(theta); % number of thetas

% compute cost function J
temp_sum = 0;
for i = 1:m
    temp_sum += -y(i) * log(sigmoid(X(i, :) * theta)) ...
        - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta));
end

J = (1 / m) * temp_sum;

% compute gradient for each theta(i)
for i = 1:n
    temp_sum = 0;
    for j = 1:m
        temp_sum += (sigmoid(X(j, :) * theta) - y(j)) * X(j, i);
    end
    grad(i) = (1 / m) * temp_sum;
end
end