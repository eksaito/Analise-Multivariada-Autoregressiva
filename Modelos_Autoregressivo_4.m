%% Modelos Autoregressivo 4
%
% Estimador de Máxima Verossimilhança
%
% Eric Kauati Saito
clc;clear;close all

N = 300;

p = 2;

v = 1;
w = sqrt(v)*randn(N,1);

b = 1;
a = [1 -1.2 0.8];

y = filter(b,a,w);

%%
LL = [];
phi = (-2:0.01:2);
for ind1 = 1:length(phi)
    for ind2 = 1:length(phi)
        LL(ind1,ind2) = MLE_AR(y,[v phi(ind1) phi(ind2)]);
    end
end
[M,I] = min(LL(:));
[I_row, I_col] = ind2sub(size(LL),I);
[phi(I_row) phi(I_col)]
mesh(LL)
%% Estimação inicial dos parametros utilizando OLS
y1 = y(1:10);
[phi_ols, var_ols] = my_ols(y1,p);
%% Estimação utilizando máxima verossimilhança
% phi_0 = [1 -1 -1];
phi_0 = [var_ols -phi_ols(2) -phi_ols(3)]; %parametros iniciais
options = optimset('PlotFcns',@optimplotfval);
fun = @(phi_hat)MLE_AR(y,phi_hat);
% phi = fminsearch(fun,phi_0,options)
A = [];
b = [];
Aeq = [];
beq = [];
lb = [-5,-5,-5];
ub = [5,5,5];
phi = fmincon(fun,phi_0,A,b,Aeq,beq,lb,ub,[],options)