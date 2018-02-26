clear
clc

addpath(genpath('./tensor_toolbox'));
rng(5489, 'twister');

m = 10;
t = 10;
n = 10;
k = 10; % rank for tensor
[X, Y] = semiBAT_data(m, t, n, k); % generate the tensor and label

[T, W] = semiBAT_fun(X, Y, k);

[~, y1] = max(Y, [], 2);
[~, y2] = max(T{4} * W, [], 2);
fprintf('accuracy %3.2e\n', sum(y1 == y2) / n);