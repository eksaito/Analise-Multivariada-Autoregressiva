%% Ljung–Box test
%
% Portmanteau Lack-of-Fit Test
%
% Input:
%   y - time series
%   p - order
% Output:
%   theta_hat - AR coefficients Estimation
%   var_hat - estimated variance of white noise inpute
% Eric Kauati Saito
% 05/2021

clc;clear all;close all

N = 300;

v = 0.3;
w = sqrt(v)*randn(N,1);

b = 1;
% a = [1 -1.2 0.8];
a = [1 0.1 -0.8 -0.27];
%  a = [1, -2.7607, 3.8106, -2.6535, 0.9238];

y = filter(b,a,w);

p = 3;
[a_ols] = my_ols(y,p);
y_ols = filter([0 -a_ols(2:end)],b,y);
e_ols = y - y_ols;
e_ols = e_ols/sqrt(var(e_ols));

maxlag = 20;
dof = maxlag - p;
[h1,p1] = lbqtest(e_ols,'Lags',maxlag,'DOF',dof)
%h2 = lbqtest(e_ols.^2,'Lags',maxlag,'DOF',dof);
%%
maxlag = 20;
dof = maxlag - p;
acf = autocorr(e_ols,'NumLags',maxlag);
acf = acf(2:end);
coef = (N-(1:maxlag))';
Q = N*(N+2)*cumsum((acf.^2)./coef);
Q = Q(maxlag);
p_value = 1- chi2cdf(Q, dof)
alpha = 0.05;
h = (alpha >= p_value)