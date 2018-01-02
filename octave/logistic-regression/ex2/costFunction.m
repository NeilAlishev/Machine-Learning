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

%% =================== Vectorized form =================

% Compute vectorized cost function J
h_values = sigmoid(X * theta);

pos = find(y == 1);
neg = find(y == 0);

cost_values_pos = - log(h_values(pos));
cost_values_neg = - log(1 - h_values(neg));

J_vectorized = (1 / m) * (sum(cost_values_pos) + sum(cost_values_neg));

fprintf('Vectorized cost function value: %f \n', J_vectorized);

% Compute vectorized gradient
grad_vectorized = (1 / m) * (X' * (h_values - y));
fprintf('Vectorized gradient values: \n');
fprintf('%f \n', grad_vectorized);

end