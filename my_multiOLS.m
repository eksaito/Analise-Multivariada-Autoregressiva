%% Multi OLS
%
% Estimação dos parametros VAR utilizando Ordinary Least Square
%
% Input:
%   y - time series
%   p - order
% Output:
%   B - AR coefficients Estimation
%   ress - residual
%   sigmau - estimated variance of white noise input
%
% Eric Kauati Saito
%
function [B, ress, sigmau, gamma, gammap, y_h] = my_multiOLS(Y,p)
[k, n] = size(Y);
Z = zeros(k*p+1,n);
Z(1,:) = ones(1,n);

for lag = 1:p
     Z(k*(lag-1)+1+1:k*(lag-1)+k+1,lag+1:end) = Y(:,1:end-lag);
end
B = Y*Z'/(Z*Z');

gamma = Z(2:end,:)*Z(2:end,:)'/n;
gammap = Z(2:end,:)*Y'/n;

sigmau = (1/(n-k*p-1))*Y*(eye(n)-Z'/(Z*Z')*Z)*Y'; %Lutkhephol - (3.2.19)

y_h = B*Z;
ress = Y-y_h;
end