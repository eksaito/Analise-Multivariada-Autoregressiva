%% MLE
%
% log Máxima Verossimilhança do AR, ressíduo -> distribuição normal
%
% Input:
%   y - time series
%   phi_0 - [Variância do ruido branco, Parâmetros do AR]
% Output:
%   LL - log Máxima Verossimilhança
%
% Eric Kauati Saito
%
function LL = MLE_AR(y,phi_0)

N = size(y,1);
var_MLE = phi_0(1);
phi_hat = phi_0(2:end);
p_hat = size(phi_hat,2);

y_hat = zeros(N,1);

y_hat(1:1+p_hat) = y(1:1+p_hat);

for t = 1+p_hat:N
    for ind_p = 1:p_hat
        y_hat(t,1) = y_hat(t,1) + phi_hat(ind_p)*y(t-ind_p);
    end
end

ress = y - y_hat;
% LL = (-N/2)*log(var(ress))-sum(ress.^2)/(2*var(ress));
LL = (-N/2)*log(var_MLE)-sum(ress.^2)/(2*var_MLE);
 LL = -LL;
end

