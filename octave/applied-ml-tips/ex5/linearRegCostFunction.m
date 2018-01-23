function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% compute all hypothesis values
h_values = X * theta;

% compute unregularized cost function value
J = (1 / (2 * m)) * sum((h_values - y).^2);

% add regularization term, don't regularize theta0
reg_theta = theta;
reg_theta(1) = 0;

J += (lambda / (2 * m)) * sum(reg_theta.^2);

% =========================================================================

% compute gradients
grad = (1 / m) * ((h_values - y)' * X);
grad = grad'; % change grad dimension

% add regularization term
grad += (lambda / m) * reg_theta;

grad = grad(:);

end
