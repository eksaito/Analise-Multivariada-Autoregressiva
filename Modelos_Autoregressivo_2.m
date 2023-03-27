%% Modelos Autoregressivo 2
%
% Estabilidade
%
% Eric Kauati Saito

%% Exemplo 2
clear;clc;close all

N = 300;
v = 0.5;
x = sqrt(v)*randn(N,1);

b = 1;
a = [1, -2.7607, 3.8106, -2.6535, 0.9238];

y = filter(b,a,x);

disp('AR')
disp(['Coef.: [' num2str(a) ']'])
disp(['Estimated Variance Input: ' num2str(v)])
disp('-----')

%% Estimação AR Yule-Walker
p = 5;

[theta_hat, var_hat] = aryule(y,p);

disp('aryule')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')

[theta_hat, var_hat] = my_ar_yw(y,p);

disp('Yule-Walker')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')
%%
figure
zplane(b,a)
isstable(b,a)

% Mdl = arima('AR',-a(2:end),'Constant',0)
% figure
% impulse(Mdl)
%% Estimação AR Ordinary Least Square
[theta_hat, var_hat] = my_ols(y,p);
disp('OLS')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')

%% Gráfico de Partial Autocorrelation Function
% Analise gráfica para estimação da ordem do modelo
p_max = 20;

figure
subplot(2,1,1)
parcorr(y, 'Method','yule-walker')
title('Yule-walker')
subplot(2,1,2)
parcorr(y, 'Method','ols')
title('OLS')