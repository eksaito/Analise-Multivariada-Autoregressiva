%% Modelos Autoregressivo 7
%
% Vector Autoregressive Model
% Testes estatísticos
%
% Eric Kauati Saito

clear;clc;close all
% rng(77)

n = 300;
k = 2;
p = 2; %ordem

x = randn(k,n+p);
A1 = [0.5, 0.1; 0.4, 0.5;];
A2 = [0, 0; 0.25, 0;];

sigma = [0.09, 0; 0, 0.04;];
R = chol(sigma);
e_n = R*randn(k,n+p);

v = [1; 3];
A = [v, A1, A2];

Z1 = ones(1,n);
Z2 = x(:,1:end-p);
Z3 = x(:,2:end-p+1);
Z = [Z1;Z2;Z3];

Y = zeros(k,n);
for t = (1+p):(n+p)
    Z(:,t) = [1; Y(:,t-1); Y(:,t-2)];
    Y(:,t) = A*Z(:,t) +e_n(:,t);
end
Y = Y(:,(1+p):end);
Z = Z(:,(1+p):end);

armairf({A1,A2},[],'Method','generalized','InnovCov',sigma);
%%
figure
subplot(2,1,1)
my_pacf(Y(1,:)',20)
subplot(2,1,2)
my_pacf(Y(2,:)',20)

%% Estimador OLS
B = Y*Z'*inv(Z*Z');
gamma = Z(2:end,:)*Z(2:end,:)'/n;
gammap = Z(2:end,:)*Y'/n;

sigma_h = (1/(n-k*p-1))*Y*(eye(n)-Z'*inv(Z*Z')*Z)*Y'; %Lutkhephol - pg 75

% b = kron(inv(Z(2:end,:)*Z(2:end,:)'),sigma_h)
y_h = B*Z;
ress = Y-y_h;

figure
subplot(2,1,1)
hold on
plot(Y(1,:))
plot(y_h(1,:))
title('Variável 1')
subplot(2,1,2)
hold on
plot(Y(2,:))
plot(y_h(2,:))
title('Variável 2')
%%
figure
subplot(2,1,1)
autocorr(ress(1,:))
subplot(2,1,2)
autocorr(ress(2,:))

%% Portmanteau Test (Whiteness of the residuals)
% Tsay - 2.7.2
% C -> Cross Correlation
maxlag = 12;

for lag = 1:maxlag
    C{lag} = ress(:,1:end-lag)*ress(:,lag+1:end)'/(n-p);
end
C0 = ress*ress'/(n-p);

for lag = 1:maxlag
    Q_lag(lag) = (trace(C{lag}'*inv(C0)*C{lag}*inv(C0)))/(n-lag);
end
Q = n^2*cumsum(Q_lag);

dof = k^2*(lag-p); %lag ou maxlag?
p_value = 1- chi2cdf(Q, dof);
alpha = 0.05;
c_value = chi2inv(1-alpha,dof);
h = (Q > c_value)
%% Teste de não normalidade
u_r = mean(ress,2);
S = (ress-u_r)*(ress-u_r)'/(n-1);
L = chol(S,'lower'); % P*inv(P) = S
wr = inv(L)*(ress-u_r);

w_s = skewness(wr');
w_k = kurtosis(wr');

lambda_s = n*(w_s*w_s')/6;
lambda_k = n*(w_k-3)*(w_k-3)'/24;
lambda_sk = lambda_s + lambda_k;
alpha = 0.1;
c_lambda = chi2inv(1-alpha,k);
c_lambda_sk = chi2inv(1-alpha,2*k);

lambda_s > c_lambda
lambda_k > c_lambda
lambda_sk > c_lambda_sk