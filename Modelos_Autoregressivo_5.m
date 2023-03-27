%% Modelos Autoregressivo 5
%
% Vector Autoregressive Model
% Matriz de covariancia
% OLS e YW
%
% Eric Kauati Saito
clear;clc;close all
% rng(77)

n = 300; 
k = 2; %no variaveis
p = 1; %ordem
T = n+p;

x = randn(k,n);
A = [0.8, 0.7; -0.4, 0.6;];
[~,D] = eig(A); %Estabilidade (|D|<1)
abs(D)
% syms z
% R = solve(det(eye(2) - A*z),z);
% raizes = vpa(R);

sigma = [4, 1; 1, 2;];
R = chol(sigma);
e_n = R*randn(k,T);

c = [5; 3]; %"nivel dc"

y = zeros(k,n+p);
for t = 1+p:T
    y(:,t) = y(:,t) + A*y(:,t-1) + c + e_n(:,t);
end
y = y(:,1+p:end);
armairf({A},[],'Method','generalized','InnovCov',sigma);
%% Matriz de Covariancia 1
p_max = 3;
[C] = xcov(y',p_max,'biased');
C = C(p_max+1:end,:);

u = mean(y,2);
S = zeros(size(y,1),size(y,1));
S_m = cell(1, p_max+1);

 for p = 0:p_max
    S = zeros(size(y,1),size(y,1));
    for va1 = 1:size(y,1)
        for va2 = 1:size(y,1)
            for lag = 1:size(y,2)-p
                S(va1,va2) = S(va1,va2) + (y(va1,lag) - u(va1))*(y(va2,lag+p) - u(va2))';
            end
        end
    end
    S_m{p+1} = S/(size(y,2)-p);
 end
% S_0 = S_m{1};
% S_m(1) = [];
S_toep = cell2mat(S_m(toeplitz(1:length(S_m))));
%% Matriz de Covariancia 2
Y = y;
[k, n] = size(Y);
Z = zeros(k*p_max,n);
Z(1,:) = ones(1,n);

for lag = 1:p_max+1 %p_max
     Z(k*(lag-1)+1:k*(lag-1)+k,lag+1:end) = Y(:,1:end-lag)-u;
end

% gamma(l) = Cov(z(t),z(t-l) = E[(z(t)-u)(z(t-l)-u)']
gamma_p = Z*Z'/(n-p_max); %gamma(p_max)
gamma_1 = (Y(:,1:end-1)-u)*(Y(:,2:end)-u)'/(n-1);%gamma(1)

% D = diag(sqrt(diag(gamma)));
D = diag(sqrt(var(Y,[],2)));
%% Estimador OLS
y = y';
Y0 = [ones(n-1,1), y(1:end-1,:)];
Y1 = y(2:end,:);
B_hat = (Y1'*Y0)/(Y0'*Y0);
C_hat = B_hat(:,1);
A_hat = B_hat(:,2:end);
ress = Y0(:,1:2)' - A_hat*Y1';
% Yt = [ones(n-1,1), y(2:end,:)];
% ress = Y0(:,2:end)' - B_hat*Yt';

disp('Coeficientes AR Originais:')
disp(A)
disp('Coeficientes AR Estimados:')
disp(A_hat)
disp('Coeficientes C Originais:')
disp(c)
disp('Coeficientes C Estimados:')
disp(C_hat)

%% Matriz de Covariancia a partir dos coeficientes
% Teorico
I = eye(k^2);
pp = kron(A,A);
dd = I - pp;
gam0 = inv(dd)*sigma(:);
gam0 = reshape(gam0,2,2); %Covariância Cruzada
gam1 = gam0*A';
D = diag(sqrt(diag(gam0)));
D0 = inv(D)*gam0*inv(D); %Correlação Cruzada
%% Yule-Walker ~ OLS
gamma_YW_0 = (Y0'*Y0)/n;
gamma_YW_1 = (Y1'*Y0)/n;

A_ = gamma_YW_1/gamma_YW_0 