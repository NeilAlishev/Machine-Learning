function [X_norm, mu, sigma] = featureNormalize(X)

% init variables
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m = size(X)(1);
num_of_features = size(X)(2);

% compute mean values for the features
for i = 1:num_of_features
    mu(i) = mean(X(:, i));
end

% compute standard deviation for the features
% Here we use standard deviation (std function), but we can also use
% range (max - min) to perform feature scaling

for i = 1:num_of_features
    sigma(i) = std(X(:, i));
end

% normalize feature values
for i = 1:m
    for j = 1:num_of_features
        X_norm(i, j) = (X(i, j) - mu(j)) / sigma(j);
    end
end
end
