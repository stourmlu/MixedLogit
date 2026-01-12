addpath('../../../Optimization/Matlab');

NumX = 8; % number of characteristics (width of X)
NumMarkets =2500;
J = 40;

X = normrnd(0,1,[J*NumMarkets NumX]);
M = 10000*ones(NumMarkets, 1);

% Heterogeneity part
idxes_heterog_coefs = [8]; % indexes of characteristics that have heterogenous coefficients
NumXhetero = length(idxes_heterog_coefs);
%%%%% Using MC integration
NumDraws = 100;
nu = normrnd(0,1,[NumDraws NumXhetero]); % NumDraws x NumXhetero
weights = 1/NumDraws * ones(NumDraws, 1); % NumDraws x 1
%%%%% Using quadrature
% TO DO
%%%%%%%%%%%%%%%%%%%%%%

NumXhomo = NumX;

Xhetero = X(:,idxes_heterog_coefs) .* reshape(nu', [1 NumXhetero NumDraws]); % (J*NumMarkets) x NumXhetero x NumDraws
Xhetero = reshape(Xhetero, [J, NumMarkets, NumXhetero, NumDraws]); % J x NumMarkets x NumXhetero x NumDraws
Xhetero = permute(Xhetero, [1 2 4 3]); % J x NumMarkets x NumDraws x NumXhetero
Xhetero = reshape(Xhetero, [J*NumMarkets*NumDraws NumXhetero]); % (J*NumMarkets*NumDraws) x NumXhetero

jm_2_mm_vec = repmat([1:(NumMarkets)], J, 1); % This is full cartesian product
jm_2_mm_vec = reshape(jm_2_mm_vec, [J*NumMarkets 1]); % (J*NumMarkets) x 1 (integers between 1 and NumMarkets)

%%% Subset it (randomly, to check robustness)
keepFlag = rand(J*NumMarkets, 1) > 0.1;
jm_2_mm_vec = jm_2_mm_vec(keepFlag); % Num_jm x 1 (values: integers between 1 and NumMarkets)
X = X(keepFlag,:); % Num_jm x NumXhomo
Xhetero = reshape(Xhetero, [J*NumMarkets, NumDraws NumXhetero]);
Xhetero = Xhetero(keepFlag,:,:);
Num_jm = size(jm_2_mm_vec, 1);
Xhetero = reshape(Xhetero, [size(Xhetero, 1)*size(Xhetero,2), NumXhetero]);

% From jm_2_mm_vec and NumDraws (given), construct jmd_2_md_vec
Num_jm = size(jm_2_mm_vec, 1);
jmd_2_mm_vec = reshape(repmat(jm_2_mm_vec, 1, NumDraws), [Num_jm*NumDraws 1]);
jmd_2_dd_vec = reshape(repmat([1:NumDraws], Num_jm, 1), [Num_jm*NumDraws 1]);
[~, ~, jmd_2_md_vec] = unique([jmd_2_dd_vec jmd_2_mm_vec], 'rows'); % (Num_jm*NumDraws) x 1

beta_true = normrnd(0,1,[NumX 1]);
sigma_true = abs(normrnd(0,0.1,[NumXhetero 1]));
params_true = [beta_true; sigma_true];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate Y
V = X * beta_true + reshape(Xhetero * sigma_true, [Num_jm, NumDraws]); % Num_jm x NumDraws
V = reshape(V, [Num_jm*NumDraws 1]); % (Num_jm*NumDraws) x 1

Vmax = accumarray(jmd_2_md_vec, V, [NumMarkets*NumDraws 1], @max); % (NumMarkets*NumDraws) x 1
V = V - Vmax(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
tmp = log(accumarray(jmd_2_md_vec, exp(V), [NumMarkets*NumDraws 1])); % (NumMarkets*NumDraws) x 1
log_pTilda = V - tmp(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
p_Tilda = reshape(exp(log_pTilda), [Num_jm, NumDraws]); % Num_jm x NumDraws

p = p_Tilda * weights; % Num_jm x 1

Y = zeros(Num_jm, 1);
for nn = 1:NumMarkets
	idces_nn = find(jm_2_mm_vec == nn); % Jn x 1
	p_nn = p(idces_nn); % Jn x 1
	M_nn = M(nn); % scalar
	Y(idces_nn) = mnrnd(M_nn, p_nn); % Jn x 1
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Estimate model
[params_star, LL_star, LL_grad, FisherInfo, params_ses] = estimate_MMNL(M, X, Y, jm_2_mm_vec, idxes_heterog_coefs, nu, weights);

disp([params_true params_star]);
%disp(table(params_true, params_star, params_ses));
