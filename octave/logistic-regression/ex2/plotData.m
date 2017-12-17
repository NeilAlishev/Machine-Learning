function plotData(X, y)

figure; hold on;

% find indices of positive and negative samples
pos = find(y==1); neg = find(y==0);

% plot positive samples
plot(X(pos, 1), X(pos, 2), 'k+', 'MarkerSize', ...
    7, 'LineWidth', 2);
% plot negative samples
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerSize', 7, ...
    'MarkerFaceColor', 'y', 'LineWidth', 2);

hold off;

end
