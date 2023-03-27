%% YULE-WALKER
%
% Estimação dos parametros AR utilizando equações de Yule-Walker
%
% Input:
%   y - time series
%   p - order
% Output:
%   theta_hat - AR coefficients Estimation
%   var_hat - estimated variance of white noise inpute
% Eric Kauati Saito
% 05/2021

function [theta_hat, var_hat] = my_ar_yw(y,p)

y = y - mean(y);

acf0 = xcorr(y,y,p,'biased')';
acf = acf0(p+1:end);
a0 = acf(1);
acf = acf./acf(1); %Normalizado r(0) = 1
R = toeplitz(acf(1:end-1)); %Matriz de autoCorrelação p x p -> r(0):r(p-1)
rxx = acf(2:end); %Vetor de autoCorrelação 1 x p -> r(1):r(p)

theta0 = -R\rxx';
theta_hat = [1 theta0'];
var_hat = a0*sum(theta_hat.*acf); 
% var_hat = acf(1) - sum((-theta0').*rxx); %Igual
end