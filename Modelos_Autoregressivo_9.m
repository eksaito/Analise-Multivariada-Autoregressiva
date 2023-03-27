%% Modelos Autoregressivo 9
%
% Vector Autoregressive Model
% Sinal real
%
% Eric Kauati Saito

clear;clc;close all

load('MVvol1_ICA.mat')
ext = squeeze(ext);
flex = squeeze(flex);
ch = [4,5,6];
epoca = 4;
sinal = squeeze(flex(epoca,ch,1*fs:5*fs))*1e6;
% sinal = squeeze(flex(epoca,ch,:));
n = size(sinal,2);
k = size(sinal,1); %no variaveis

figure
subplot(3,1,1)
plot(sinal(1,:))
subplot(3,1,2)
plot(sinal(2,:))
subplot(3,1,3)
plot(sinal(3,:))
%% Relação ordem vs nº amostras
p0 = 1;
pmax = 30;
for p = p0:pmax
    [B{p}, ress, sigmau{p}] = my_multiOLS(sinal,p);
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
%%
clc
p = 2;
A0 = ones(k,k*p);
c = ones(k,1);
B_0 = [c,A0];
LL = MLE_var(sinal,B_0);
%%
options = optimset('PlotFcns',@optimplotfval);
fun = @(B_0)MLE_var(sinal,B_0);
% phi = fminsearch(fun,phi_0,options)
a = [];
b = [];
Aeq = [];
beq = [];
% lb = [-5,-5,-5];
% ub = [5,5,5];
B_hat = fmincon(fun,B_0,a,b,Aeq,beq,[],[],[],options)
%% Analise de Estacionariedade
% result = 0: "stationary" result = 1: "no stationary"
alfa = 5;
[~,~,~,result_r1] = run_test(sinal(1,:),alfa);
[~,~,~,result_r2] = run_test(sinal(2,:),alfa);
[~,~,~,result_r3] = run_test(sinal(3,:),alfa);
[~,~,~,result_a1] = revarr_test(sinal(1,:),alfa);
[~,~,~,result_a2] = revarr_test(sinal(2,:),alfa);
[~,~,~,result_a3] = revarr_test(sinal(3,:),alfa);
display('Resultados run test')
display([result_r1, result_r2, result_r3]) 
display('Resultados revarr test')
display([result_a1, result_a2, result_a3])
%%
figure
subplot(2,1,1)
autocorr(sinal(1,:))
subplot(2,1,2)
parcorr(sinal(1,:))

figure
subplot(2,1,1)
autocorr(sinal(2,:))
subplot(2,1,2)
parcorr(sinal(2,:))

figure
subplot(2,1,1)
autocorr(sinal(3,:))
subplot(2,1,2)
parcorr(sinal(3,:))