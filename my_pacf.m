%% PLOT Partial Autocorrelation Function
%
% Grafico de Partial Autocorrelation Function do modelo AR (utilizando
% Yule-Walker)
%
% Input:
%   y - time series
%   p_max - max lag
% Output:
%
% Eric Kauati Saito
% 05/2021

function my_pacf(y,p_max)

pacf = zeros(p_max+1,1);
lags = 0:p_max;
pacf(1) = 1;
for p_ind = 1:p_max
    theta_hat = my_ar_yw(y,p_ind);
    pacf(p_ind+1) =  -theta_hat(p_ind+1);
end

conf = 0.95;
critvalue = sqrt(2)*erfinv(conf);
upconf = critvalue/sqrt(length(y));
lowconf = -critvalue/sqrt(length(y));

stem(lags,pacf)
line([0,p_max],[upconf,upconf],'Color','red','LineStyle','--')
line([0,p_max],[lowconf,lowconf],'Color','red','LineStyle','--')
xlabel('Lag')
end