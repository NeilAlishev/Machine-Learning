function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
num_of_features = size(X, 2);

cost_func_pds = zeros(num_of_features, 1); % array for the partial derivatives

for iter = 1:num_iters
    temp_sum = zeros(num_of_features, 1); % array for the temporary sum values

    % we do a batch gradient descent, so we iterate over all samples
    for i = 1:m
        % TODO: refactor this using vectorization

        error_value = (X(i, :) * theta - y(i));

        for j = 1:num_of_features
            temp_sum(j) += error_value * X(i, j);
        end
    end

    % count partial derivatives
    for i = 1:num_of_features
        cost_func_pds(i) = (1 / m) * temp_sum(i);
    end

    % update theta values
    for i = 1:num_of_features
        theta(i) -= alpha * cost_func_pds(i);
    end

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
end
end
