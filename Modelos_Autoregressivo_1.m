%% Modelos Autoregressivo 1
%
% Modelagem AR
%
% Eric Kauati Saito

%% Exemplo 1
clear;clc;close all

N = 300;
v = 0.3;
x = sqrt(v)*randn(N,1);

b = 1;
a = [1 -1.2 0.8];

y = filter(b,a,x);
disp('AR')
disp(['Coef.: [' num2str(a) ']'])
disp(['Estimated Variance Input: ' num2str(v)])
disp('-----')
%% Gr�fico de Partial Autocorrelation Function
% Analise gr�fica para estima��o da ordem do modelo
p_max = 20;

figure
subplot(2,1,1)
my_pacf(y,p_max)
title('My Partial Autocorrelation Function')
subplot(2,1,2)
parcorr(y, 'Method','yule-walker')

%% Estima��o AR Yule-Walker
p = 2;
[theta_hat, var_hat] = my_ar_yw(y,p);

disp('Yule-Walker')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')
%% Estima��o AR Ordinary Least Square
[theta_hat, var_hat] = my_ols(y,p);
disp('OLS')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')