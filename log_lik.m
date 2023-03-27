function val = log_lik(theta,data)
% n = exp(theta);
val = -sum(log(normpdf(data,0,theta)));
end