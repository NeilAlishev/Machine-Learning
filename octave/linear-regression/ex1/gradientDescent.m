function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp_sum1 = 0; % temp sum for the partial derivative of theta1
    temp_sum2 = 0; % temp sum for the partial derivative of theta2

    % we do a batch gradient descent, so we iterate over all samples
    for i = 1:m
        % TODO: refactor this using vectorization

        error_value = (X(i, :) * theta - y(i));

        temp_sum1 += error_value; % x1 is always equal to 1
        % NOTE: for multivariable linear regerssion there will be many such samples
        temp_sum2 += error_value * X(i,2); % every x2 sample
    end

    % these are two partial derivatives of cost function J
    % (d / d(theta1)) and (d / d(theta2))
    cost_func_pd1 = (1 / m) * temp_sum1;
    cost_func_pd2 = (1 / m) * temp_sum2;

    % update theta values
    theta(1) = theta(1) - alpha * cost_func_pd1;
    theta(2) = theta(2) - alpha * cost_func_pd2;

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
end

end
