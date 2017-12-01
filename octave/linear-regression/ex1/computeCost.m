function J = computeCost(X, y, theta)

m = length(y); % number of training examples

% TODO: refactor this using vectorization
J = 0;
temp_sum = 0;
for i = 1:m
    temp_sum += (X(i, :) * theta - y(i))^2;
end

J = (1 / (2 * m)) * temp_sum;

end
