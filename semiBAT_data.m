function [X, Y] = semiBAT_data(m, t, n, k)

B = randn(m,k);
T = randn(t,k);
A = randn(n,k);
X = ktensor({B,B,T,A});

Y = zeros(n,2);
l = ceil(n/2);
Y(1:l,1) = 1;
Y(l+1:end,2) = 1;

X = tensor(X);