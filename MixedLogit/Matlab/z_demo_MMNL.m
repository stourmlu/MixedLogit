addpath('../../../Optimization/Matlab');

NumX = 15; % number of characteristics (width of X)
NumMarkets = 500;
J = 40;

X = normrnd(0,1,[J*NumMarkets NumX]);
M = 100*ones(NumMarkets, 1);

% Heterogeneity part
idxes_heterog_coefs = [14 15]; % indexes of characteristics that have heterogenous coefficients
NumXhetero = length(idxes_heterog_coefs);
%%%%% Using MC integration
NumDraws = 3;
nu = normrnd(0,1,[NumDraws NumXhetero]); % NumDraws x NumXhetero
weights = 1/NumDraws * ones(NumDraws, 1); % NumDraws x 1
%%%%% Using quadrature
% TO DO
%%%%%%%%%%%%%%%%%%%%%%

Xhomo = X; % (J*NumMarkets) x NumXhomo
NumXhomo = NumX;

Xhetero = X(:,idxes_heterog_coefs) .* reshape(nu', [1 NumXhetero NumDraws]); % (J*NumMarkets) x NumXhetero x NumDraws
Xhetero = reshape(Xhetero, [J, NumMarkets, NumXhetero, NumDraws]); % J x NumMarkets x NumXhetero x NumDraws
Xhetero = permute(Xhetero, [1 2 4 3]); % J x NumMarkets x NumDraws x NumXhetero
Xhetero = reshape(Xhetero, [J*NumMarkets*NumDraws NumXhetero]); % (J*NumMarkets*NumDraws) x NumXhetero

nn_vec = repmat([1:NumMarkets*NumDraws], J, 1); % This is full cartesian product
nn_vec = reshape(nn_vec, [J*NumMarkets*NumDraws, 1]); % (J*NumMarkets*NumDraws) x 1 (integers between 1 and NumMarkets*NumDraws)

nn_vec2 = repmat([1:(NumMarkets)], J, 1); % This is full cartesian product
nn_vec2 = reshape(nn_vec2, [J*NumMarkets 1]); % (J*NumMarkets) x 1 (integers between 1 and NumMarkets)

beta_true = normrnd(0,1,[NumX 1]);
sigma_true = normrnd(0,1,[NumXhetero 1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate Y
V = reshape(X * beta_true, [J NumMarkets]) + reshape(Xhetero * sigma_true, [J NumMarkets, NumDraws]); % J x NumMarkets x NumDraws
V = reshape(V, [J*NumMarkets*NumDraws 1]); % (J*NumMarkets*NumDraws) x 1

Vmax = accumarray(nn_vec, V, [NumMarkets*NumDraws 1], @max); % (NumMarkets*NumDraws) x 1
V = V - Vmax(nn_vec); % (J*NumMarkets*NumDraws) x 1
tmp = log(accumarray(nn_vec, exp(V), [NumMarkets*NumDraws 1])); % (NumMarkets*NumDraws) x 1
log_pTilda = V - tmp(nn_vec); % (J*NumMarkets*NumDraws) x 1
p_Tilda = reshape(exp(log_pTilda), [J*NumMarkets, NumDraws]); % (J*NumMarkets) x NumDraws

p = p_Tilda * weights; % (NumMarkets*J) x 1

Y = zeros(NumMarkets*J, 1);
for nn = 1:NumMarkets
	idces_nn = find(nn_vec2 == nn); % Jn x 1
	p_nn = p(idces_nn); % Jn x 1
	M_nn = M(nn); % scalar
	Y(idces_nn) = mnrnd(M_nn, p_nn); % Jn x 1
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Estimate model
%[beta_star, LL_star, LL_grad, FisherInfo, beta_ses] = estimate_MNL(M, X, Y, nn_vec, nu, weights);
%
%disp(table(beta_true, beta_star, beta_ses));
