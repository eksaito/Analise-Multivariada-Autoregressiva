%% Modelos Autoregressivo 6
%
% Vector Autoregressive Model 
% Ordem 2
%
% Eric Kauati Saito
clear;clc;close all
% rng(77)

n = 300;
k = 2; %no variaveis
p = 2; %ordem

x = randn(k,n);
% Modo 1
A1 = [0.5, 0.1; 0.4, 0.5;];
A2 = [0, 0; 0.25, 0;];
A = {A1,A2};
sigma = [0.09, 0; 0, 0.04;];
R = chol(sigma);
e_n = R*randn(k,n+p);

c = [0; 0];

y = zeros(k,n+p);

for t = (1+p):(n+p)
    for p_ind = 1:p
        y(:,t) = y(:,t) + A{p_ind}*y(:,t-p_ind);
    end
    y(:,t) = y(:,t) + c + e_n(:,t);
end
y = y(:,1+p:end);

% Modo 2
v = [c; zeros(p,1)];
A_2 = [A{1} A{2}; eye(k) zeros(p)];
e_n2 = [e_n; zeros(p,n+p)];

y2 = zeros(k*p,n+p);
for t = (1+p):(n+p)
    y2(:,t) = v + A_2*y2(:,t-1) + e_n2(:,t);
end
y2 = y2(:,1+p:end);

% Resposta ao impulso
armairf({A1 A2},[],'Method','generalized','InnovCov',sigma);
%% Estimador OLS
% Modo 1
y = y';
Y0 = [ones(n-1,1), y(1:end-1,:),[zeros(1,k); y(1:end-2,:)]];
Y1 = [y(2:end,:), y(1:end-1,:)];
B_hat = (Y1'*Y0)/(Y0'*Y0);
C_hat = B_hat(:,1);
A_hat = B_hat(:,2:end);
y_hat = A_hat(1:2,:)*Y1';
ress = Y0(:,2:3)' - A_hat(1:2,:)*Y1' + C_hat(1:2);

% Modo 2
y2 = y2'; %[y(t) y(t+1)]
Y02 = [ones(n-1,1), y2(1:end-1,:)];
Y12 = y2(2:end,:);
B_hat2 = (Y12'*Y02)/(Y02'*Y02);
C_hat2 = B_hat2(:,1);
A_hat2 = B_hat2(:,2:end);

figure
subplot(2,1,1)
hold on
plot(y(1:end-1,1)')
plot(y_hat(1,:))
title('Variável 1')
subplot(2,1,2)
hold on
plot(y(1:end-1,2)')
plot(y_hat(2,:))
title('Variável 2')

disp('Coeficientes AR Originais:')
disp([A1 A2])
disp('Coeficientes AR Estimados:')
disp(A_hat)
disp('Coeficientes C Originais:')
disp(c)
disp('Coeficientes C Estimados:')
disp(C_hat)
%%
% A_hat3 = S_m{2}*inv(S_m{1})
%  cell2mat(S_m0)*inv(S_toep)