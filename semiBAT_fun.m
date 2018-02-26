function [Xba, W] = semiBAT_fun(X, Y, k)
% semiBAT performs semi-supervised brain network analysis based on 
% constrained tensor factorization

% INPUT
% X: brain networks stacked in a 4-way tensor
% Y: label information
% k: rank of CP factorization
%
% OUTPUT
% Xba is the factor tensor containing
%   vertex factor matrix B = Xba{1},
%   temporal factor matrix T = Xba{3} and
%   graph factor matrix A = Xba{4}
% W is the weight matrix
%
% Example: see semiBAT_demo.m
%
% Reference:
% Bokai Cao, Chun-Ta Lu, Xiaokai Wei, Philip S. Yu and Alex D. Leow. 
% Semi-supervised Tensor Factorization for Brain Network Analysis. 
% In ECML/PKDD 2016.
%
% Dependency:
% Matlab tensor toolbox v 2.6
% Brett W. Bader, Tamara G. Kolda and others
% http://www.sandia.gov/~tgkolda/TensorToolbox

%% set algorithm parameters
alpha = 0.1; % weight for classification loss
lambda = 0.1; % weight for regularization

printitn = 1;
maxiter = 200;
fitchangetol = 1e-4;

upsilon = 1e-6;
phi = 1e-6;
psi = 1e-6;
upsilonmax = 1e6;
phimax = 1e6;
psimax = 1e6;
rho = 1.15;

%% compute statistics
dim = size(X);
normX = norm(X);
numClass = size(Y, 2);
m = dim(1);
t = dim(3);
n = dim(4);
l = size(Y, 1);
D = [eye(l), zeros(l,n-l)];

%% initialization
B = rand(m,k);
P = B;
T = rand(t,k);
A = rand(n,k);
A = orth(A);
Q = A;
W = randn(k,numClass);
% Lagrange multipliers
Upsilon = zeros(m,k);
Phi = zeros(n,k);
Psi = zeros(k,k);

%% main loop
fit = 0;
for iter = 1: maxiter
    fitold = fit;
    % update B
    ete = (P' * P) .* (T' * T) .* (A' * A); % compute E'E
    b = 2 * ete + upsilon * eye(k);
    c = 2 * mttkrp(X,{B,P,T,A},1) + upsilon * P + Upsilon;
    B = c / b;

    % update P
    ftf = (B' * B) .* (T' * T) .* (A' * A); % compute F'F
    b = 2 * ftf + upsilon * eye(k);
    c = 2 * mttkrp(X,{B,P,T,A},2) + upsilon * B - Upsilon;
    P = c / b;
    
    % update T
    gtg = (B' * B) .* (P' * P) .* (A' * A); % compute G'G
    b = 2 * gtg;
    c = 2 * mttkrp(X,{B,P,T,A},3);
    T = c / b;
    
    % update A
    hth = (B' * B) .* (P' * P) .* (T' * T); % compute H'H
    a = psi * (Q * Q'); 
    b = 2 * hth + 2 * alpha * W * W' + phi * eye(k);
    c = 2 * mttkrp(X,{B,P,T,A},4) + 2 * alpha * D' * Y * W' + (phi+ psi) * Q + Phi - Q * Psi;
    A = lyap(a,b,-c);
    
    % update Q
    b = psi * A * A' + phi * eye(n);
    c = (phi + psi) * A - Phi - A * Psi';
    Q = b \ c;
    
    % update W
    while true
        W_old = W;
        DA = D * A;
        Omega = 1 ./ (2 * sqrt(sum((W).^2,1) + 1e-6));
        a = DA' * DA;
        b = lambda * diag(Omega);
        c = DA' * Y;
        W = lyap(a,b,-c);
        if (norm(W - W_old) < 1e-6)
            break;
        end
    end;
    
    % update Lagrange multipliers
    Upsilon = Upsilon + upsilon * (P - B);
    Phi = Phi + phi * (Q - A);
    Psi = Psi + psi * (Q' * A - eye(k));
    
    % update penalty parameters
    upsilon = min(rho * upsilon, upsilonmax);
    phi = min(rho * phi, phimax);
    psi = min(rho * psi, psimax);
    
    % compute the fit
    Xba = ktensor({B,P,T,A});
    normresidual = sqrt(normX^2 + norm(Xba)^2 - 2 * innerprod(X,Xba));
    fit = 1 - (normresidual / normX);
    fitchange = abs(fitold - fit);
    
    if mod(iter,printitn)==0
        fprintf(' Iter %2d: fitdelta = %7.1e\n', iter, fitchange);
    end
    
    % check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        break;
    end
end

%% clean up final results
Xba = arrange(Xba); % columns are normalized

fprintf('fit %3.2e\n', fit);

end