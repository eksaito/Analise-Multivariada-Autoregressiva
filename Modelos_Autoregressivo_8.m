%% Modelos Autoregressivo 8
%
% Vector Autoregressive Model
% Seleção da ordem
%
% Eric Kauati Saito

clear;clc;close all
% rng(77)

n = 300;
k = 2; %no variaveis
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
Z0 = [Z1;Z2;Z3];

Y = zeros(k,n);
for t = (1+p):(n+p)
    Z0(:,t) = [1; Y(:,t-1); Y(:,t-2)];
    Y(:,t) = A*Z0(:,t) +e_n(:,t);
end
Y = Y(:,(1+p):end);
clear Z1 Z2 Z3 Z0
%%
p0 = 1;
pmax = 5;
for p = p0:pmax
    [B{p}, ress, sigmau{p}] = my_multiOLS(Y,p);
    % Matlab
%     AIC(p) = n*log(det(ress*ress'/n)) + 2*p + n*(k*(log(2*pi)+1));
%     BIC(p) = n*log(det(ress*ress'/n)) + p*log(n) + n*(k*(log(2*pi)+1));
%     fpe(p) = det(ress*ress'/n) * ((1+p/n)/(1-p/n));
    % Lutkephohl and Tsay
    AIC(p) = log(det(ress*ress'/n)) + (2*p*k^2)/n;
    BIC(p) = log(det(ress*ress'/n)) + (p*k^2)*log(n)/n;
    HQ(p) = log(det(ress*ress'/n)) + (2*p*k^2)*log(log(n))/n;
    fpe(p) = ((n+k*p+1)/(n-k*p-1))^k * det(ress*ress'/n);
    
end

%%
[M_AIC,I_AIC] = min(AIC);
[M_BIC,I_BIC] = min(BIC);
[M_HQ,I_HQ] = min(HQ);
[M_fpe,I_fpe] = min(fpe);

figure
subplot(4,1,1)
hold on
plot(AIC)
plot(I_AIC,M_AIC,'r*')
ylabel('AIC')
subplot(4,1,2)
hold on
plot(BIC)
plot(I_BIC,M_BIC,'r*')
ylabel('BIC')
subplot(4,1,3)
hold on
plot(HQ)
plot(I_HQ,M_HQ,'r*')
ylabel('HQ')
subplot(4,1,4)
hold on
plot(fpe)
plot(I_fpe,M_fpe,'r*')
ylabel('fpe')
%% Modified likelihood ratio test statistic
% Tsay - 2.6.1
% p+1 ~ p
for p = p0+1:pmax
   M_lik(p) = -(n-pmax-1.5-k*p)*log(det(sigmau{p})/det(sigmau{p-1}));
end

dof = k^2;
p_value = 1- chi2cdf(M_lik, dof);
alpha = 0.05;
c_value = chi2inv(1-alpha,dof);
(M_lik > c_value)
