function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%   regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples

% Compute vectorized cost function J
h_values = sigmoid(X * theta);

pos = find(y == 1);
neg = find(y == 0);

cost_values_pos = - log(h_values(pos));
cost_values_neg = - log(1 - h_values(neg));

J = (1 / m) * (sum(cost_values_pos) + sum(cost_values_neg)) + ...
    (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

% Compute vectorized gradient
grad = (1 / m) * (X' * (h_values - y));
temp = theta;
temp(1) = 0; % because we don't add anything for j = 0
grad = grad + (lambda / m) * temp;

end
