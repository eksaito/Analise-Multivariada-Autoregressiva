%% Ordinary Least Square
%
% Estimação dos parametros AR utilizando Ordinary Least Square
%
% Input:
%   y - time series
%   p - order
% Output:
%   theta_hat - AR coefficients Estimation
%   var_hat - estimated variance of white noise inpute
%
% Eric Kauati Saito
% 05/2021

function [theta_hat, var_hat] = my_ols(y,p)

y = y - mean(y);
x = zeros(length(y)-p,p);

for ind_p = 1:p
    x(:,ind_p) = y(p-(ind_p-1):end-(ind_p));
end
y = y(p+1:end);

theta0 = (x'*x)\x'*y;

var_hat = (y'*y - theta0'*x'*x*theta0)/(length(y)-(p+1));
theta_hat = [1 -theta0'];
end