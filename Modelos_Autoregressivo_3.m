%% Modelos Autoregressivo 3
%
% residuo/erro de predição e escolha de ordem
%
% Eric Kauati Saito
clc;clear all;close all

N = 300;

p = 2;

v = 0.3;
w = sqrt(v)*randn(N,1);

b = 1;
% a = [1 -1.2 0.8];
a = [1, -2.7607, 3.8106, -2.6535, 0.9238];

disp('AR')
disp(['Coef.: [' num2str(a) ']'])
disp(['Estimated Variance Input: ' num2str(v)])
disp('-----')


y = filter(b,a,w);
% OLS
y = y - mean(y);

x = zeros(length(y)-p,p);

for ind_p = 1:p
    x(:,ind_p) = y(p-(ind_p-1):end-(ind_p));
end

y = y(p+1:end);

theta0 = (x'*x)\x'*y;
theta_hat = [1 -theta0'];
var_hat = (y'*y - theta0'*x'*x*theta0)/(length(y)-(p+1));

disp('OLS')
disp(['Coef.: [' num2str(theta_hat) ']'])
disp(['Estimated Variance Input: ' num2str(var_hat)])
disp('-----')

y_yw = x*theta0; %predição
e = y - y_yw; %residuos
%% x*theta0 vs Filtro AR->MA
close all
y_hat2 = filter([0 theta0'],[1],y);
figure
hold on
plot(y_yw)
plot(y_hat2)
%% Plot
close all
figure
hold on
plot(y,'k')
plot(y_yw,'r-.')
legend('Sinal Original', 'Sinal Predito')
title('Sinal Original x Sinal Predito')

figure
plot(e)
title('Resíduo')

figure
histogram(e)
title('Histograma do resíduo')
%%
Mdl = arima('AR',theta0,'Constant',0);
EstMdl = estimate(Mdl,y)
E = infer(EstMdl,y);
%%
figure;
hold on
plot(e)
plot(E)
title 'Inferred Residuals'; 


%% Escolha da Ordem AIC/BIC
clc;clear all;close all

N = 300;

v = 0.3;
w = sqrt(v)*randn(N,1);

b = 1;
% a = [1 -1.2 0.8];
 a = [1 0.1 -0.8 -0.27];
%  a = [1, -2.7607, 3.8106, -2.6535, 0.9238];

disp('AR')
disp(['Coef.: [' num2str(a) ']'])
disp(['Estimated Variance Input: ' num2str(v)])
disp('-----')

y = filter(b,a,w);

p_max = 20;
AIC_yw = zeros(p_max,1);
BIC_yw = zeros(p_max,1);

for p = 1:p_max
[a_yw] = aryule(y,p);
[a_ols] = my_ols(y,p);

y_yw = filter([0 -a_yw],b,y);
y_ols = filter([0 -a_ols],b,y);

e_yw = y - y_yw;
e_ols = y - y_ols;

AIC_yw(p) = N*log(e_yw'*e_yw/N) + 2*p; 
BIC_yw(p) = N*log(e_yw'*e_yw/N) + p*log(N);
AIC_ols(p) = N*log(e_ols'*e_ols/N) + 2*p; 
BIC_ols(p) = N*log(e_ols'*e_ols/N) + p*log(N);
% Outros
end
disp('YuleWalker - Order:')
disp(['AIC: ' num2str(find(AIC_yw == min(AIC_yw)))])
disp(['BIC: ' num2str(find(BIC_yw == min(BIC_yw)))])
disp('OLS - Order:')
disp(['AIC: ' num2str(find(AIC_ols == min(AIC_ols)))])
disp(['BIC: ' num2str(find(BIC_ols == min(BIC_ols)))])
disp('-----')
% Pesquisar 'relative likelihood'
%% Estabilidae Exemplo 3
figure
zplane(b,a)
isstable(b,a)
%%
p = 2;
[a_ols] = my_ols(y,p);
y_ols = filter([0 -a_ols(2:end)],b,y);
e_ols = y - y_ols;
%%
stdr = e_ols/sqrt(var(e_ols));

figure
subplot(2,2,1)
plot(stdr)
title('Standardized Residuals')
subplot(2,2,2)
histogram(stdr,10)
title('Histogram')
subplot(2,2,3)
autocorr(stdr)
subplot(2,2,4)
qqplot(stdr)

lags = 20;
dof = lags - p;
%H0 = Test the hypothesis that the residual series is not autocorrelated
%alpha 0.05
 h1 = lbqtest(stdr,'Lags',lags,'DOF',dof)
 h2 = lbqtest(stdr.^2,'Lags',lags,'DOF',dof)

