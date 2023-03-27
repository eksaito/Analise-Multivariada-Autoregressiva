%% Modelos Autoregressivo 2.b
%
%
% Autocorrelação vs Plano Z
%   
%
% Eric Kauati Saito

clc;clear all;close all

N = 300;

v = 0.3;
w = sqrt(v)*randn(N,1);

b = 1;
a1 = [1 -1.2 0.8];
a2 = [1 0.1 -0.8 -0.27];
a3 = [1, -2.7607, 3.8106, -2.6535, 0.9238];
y1 = filter(b,a1,w);
y2 = filter(b,a2,w);
y3 = filter(b,a3,w);
%% Exemplo 1
p = 20;

figure(1)
subplot(2,2,1)
[r,lags] = xcorr(y1,p,'coeff');
stem(lags(p+1:end),r(p+1:end))
title('Autocorrelation')
subplot(2,2,3)
parcorr(y1, 'Method','yule-walker')
subplot(2,2,[2 4])
zplane(b,a1)
%% Exemplo 2
p = 20;

figure(2)
subplot(2,2,1)
[r,lags] = xcorr(y2,p,'coeff');
stem(lags(p+1:end),r(p+1:end))
title('Autocorrelation')
subplot(2,2,3)
parcorr(y2, 'Method','yule-walker')
subplot(2,2,[2 4])
zplane(b,a2)
%% Exemplo 3
p = 20;

figure(3)
subplot(2,2,1)
[r,lags] = xcorr(y3,p,'coeff');
stem(lags(p+1:end),r(p+1:end))
title('Autocorrelation')
subplot(2,2,3)
parcorr(y3, 'Method','yule-walker')
subplot(2,2,[2 4])
zplane(b,a3)