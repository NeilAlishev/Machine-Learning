function g = sigmoid(z)

% Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
g = arrayfun(@(x) 1 / (1 + exp(-x)), z);

end