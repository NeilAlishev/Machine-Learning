function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples
n = length(theta); % number of thetas

% compute cost function J
temp_sum = 0;
for i = 1:m
    temp_sum += -y(i) * log(sigmoid(X(i, :) * theta)) ...
        - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta));
end

J = (1 / m) * temp_sum;

% compute regularization term
regularization_term = 0;
for i = 1:n
    regularization_term += theta(2)^2;
end

regularization_term *= lambda / (2 * m);

% compute final cost
J += regularization_term;

% compute gradient for theta(1)
temp_sum = 0;
for j = 1:m
    temp_sum += (sigmoid(X(j, :) * theta) - y(j)) * X(j, 1);
end
grad(1) = (1 / m) * temp_sum;

% compute gradient for each other theta(i)
for i = 2:n
    temp_sum = 0;
    for j = 1:m
        temp_sum += (sigmoid(X(j, :) * theta) - y(j)) * X(j, i);
    end
    grad(i) = ((1 / m) * temp_sum) + ((lambda / m) * theta(i));
end

end
