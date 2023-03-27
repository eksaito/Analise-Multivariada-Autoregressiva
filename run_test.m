function [r,rsup,rinf,result] = run_test(x,alfa);
% [r,rsup,rinf,result] = run_test(x,alfa);
% Computes the run test according to Bendat & Piersol 1986 pp.  95-97
% input parameters
%   x       => signal to be tested
%   alfa    => % significance level
% output parameters
%   r       => run value
%   rsup    => superior r value (alfa/2)
%   rinf    => inferior r value  (1-alfa/2)
%   result  =>  result = 0: "stationary" result = 1: "no stationary"
N = length(x);
mu_r = N/2 + 1;
var_r = N.*(N-2)./(4*(N-1));
rsup = fix(norminv(1-alfa/100/2,mu_r,sqrt(var_r)));
rinf = fix(norminv(alfa/100/2,mu_r,sqrt(var_r)));

x_median = median(x);
vet = zeros(size(x));
ind = find(x>x_median);
vet(ind) = 1;
r = 1;
for i =1:length(x)-1
    if vet(i+1) ~= vet(i)
        r = r+1;
    end
end

if (r>rinf) & (r<=rsup)
    result = 0;
else
    result = 1;
end


