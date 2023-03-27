function [A,Asup,Ainf,result] = revarr_test(x,alfa);
% [A,Asup,Ainf,result] = revarr_test(x,alfa);
%Computes the reverse arrangements test according to Bendat & Piersol 1986
% pp. 97-99
% input parameters
%   x       => signal to be tested
%   alfa    => % significance level
% output parameters
%   A       => number of reverse arrangements
%   Asup    => superior A value (alfa/2)
%   Ainf    => inferior A value  (1-alfa/2)
%   result  =>  result = 0: "stationary" result = 1: "no stationary"
N = length(x);
mu_A =  N*(N-1)/4;
var_A = N*(2*N+5)*(N-1)/72;
Asup = fix(norminv(1-alfa/100/2,mu_A,sqrt(var_A)));
Ainf = fix(norminv(alfa/100/2,mu_A,sqrt(var_A)));

for i = 1:N-1,
    A(i) = length(find(x(i)>x(i+1:end)));
end
A = sum(A);

if (A>Ainf) & (A<=Asup)
    result = 0;
else
    result = 1;
end


