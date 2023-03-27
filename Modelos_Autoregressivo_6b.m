%% Modelos Autoregressivo 6.b
%
% Vector Autoregressive Model 
% Estimador de Máxima Verossimilhança
%
% Eric Kauati Saito
clear;clc;close all
rng(77)

n = 300;
k = 2; %no variaveis
p = 2; %ordem

x = randn(k,n+p);
A1 = [0.5, 0.1; 0.4, 0.5;];
A2 = [0, 0; 0.25, 0;];

sigma = [0.09, 0; 0, 0.04;];
R = chol(sigma);
e_n = R*randn(k,n+p);

v = [4; 3];
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
clc
A0 = ones(k,k*p);
c = ones(k,1);
B_0 = [c,A0];
LL = MLE_var(Y,B_0);
%%
options = optimset('PlotFcns',@optimplotfval);
fun = @(B_0)MLE_var(Y,B_0);
% phi = fminsearch(fun,phi_0,options)
a = [];
b = [];
Aeq = [];
beq = [];
% lb = [-5,-5,-5];
% ub = [5,5,5];
B_hat = fmincon(fun,B_0,a,b,Aeq,beq,[],[],[],options)