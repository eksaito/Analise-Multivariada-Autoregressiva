%% MLE
%
% log Máxima Verossimilhança do var, ressíduo -> distribuição normal
%
% Input:
%   y - time series
%   B_0 - [c, Parâmetros do AR]
% Output:
%   LL - log Máxima Verossimilhança
%
% Eric Kauati Saito
%
function LL = MLE_var(Y,B_0)
[k, n] = size(Y);
p = (size(B_0,2) - 1)/k;
% sigmau = B_0(:,1:k);
% B = B_0(:,k+1:end);
cte = 0;
Z = zeros(k*p+1,n);
Z(1,:) = ones(1,n);



for lag = 1:p
     Z(k*(lag-1)+1+1:k*(lag-1)+k+1,lag+1:end) = Y(:,1:end-lag);
end

Y_hat = B_0*Z;
ress = Y - Y_hat;
% LL = (-N/2)*log(det(sigmau))-sum(ress.^2)/(2*det(sigmau));
% LL = cte - (n-p)/2 * log(det(sigmau)) - (1/2)*trace(inv(sigmau)*(ress*ress'));
sigmau = (ress*ress')/(n-p);
Sb = trace(ress'*inv(sigmau)*ress);
LL = cte - (n-p)/2 * log(det(sigmau)) - (1/2)*Sb; % Tsay Eq.(2.39)
LL = -LL;
end