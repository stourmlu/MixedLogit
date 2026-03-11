addpath('../../../Optimization/Matlab');

%%%%%%%%%%%%%%%%%%%%%%%%%%%% GENERATE X %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumX = 8; % number of characteristics (width of X)
NumMarkets =2500;
J = 20;

% Heterogeneity part
idxes_heterog_coefs = [7 8]; % indexes of characteristics that have heterogenous coefficients
NumXhetero = length(idxes_heterog_coefs);

X = normrnd(0,1,[J*NumMarkets NumX]);
M = 100*ones(NumMarkets, 1);

% Subset (randomly) the rows of X so that the number of options is not the same for all markets
keepFlag = rand(J*NumMarkets, 1) > 0.1;
jm_2_mm_vec = repmat([1:(NumMarkets)], J, 1); % This is full cartesian product
jm_2_mm_vec = reshape(jm_2_mm_vec, [J*NumMarkets 1]); % (J*NumMarkets) x 1 (integers between 1 and NumMarkets)
jm_2_mm_vec = jm_2_mm_vec(keepFlag); % Num_jm x 1 (values: integers between 1 and NumMarkets)
X = X(keepFlag,:); % Num_jm x NumXhomo
Num_jm = size(jm_2_mm_vec, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% GENERATE TRUE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_true = normrnd(0,1,[NumX 1]);
sigma_true = abs(normrnd(0,0.5,[NumXhetero 1]));
params_true = [beta_true; sigma_true];

%%%%%%%%%%%%%%%%%%%%%%%%%%%% GENERATE Y %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% DGP 1 %%%%%%%%%%
Y = zeros(Num_jm, 1);
for nn = 1:NumMarkets
	idces_nn = find(jm_2_mm_vec == nn); % Jn x 1
	Jn = length(idces_nn);
	M_nn = M(nn); % scalar
	nu = normrnd(0,1,[M_nn NumXhetero]); % M_nn x NumXhetero
	X_nn = X(idces_nn,:); % Jn x NumX
	Xhetero_nn = reshape(X_nn(:,idxes_heterog_coefs), [Jn 1 NumXhetero]) .* reshape(nu, [1 M_nn NumXhetero]); % Jn x M_nn x NumXhetero
	Xhetero_nn = reshape(Xhetero_nn, [Jn*M_nn NumXhetero]); % (Jn*M_nn) x NumXhetero
	
	V_nn = X_nn * beta_true + reshape(Xhetero_nn * sigma_true, [Jn M_nn]); % Jn x M_nn
	V_nn = V_nn - max(V_nn, [], 1); % Jn x M_nn
	eV_nn = exp(V_nn); % Jn x M_nn
	pTilda_nn = eV_nn ./sum(exp(V_nn), 1); % Jn x M_nn
	
	Y_nn = zeros(1, Jn);
	for ss = 1:M_nn
		a = mnrnd(1, pTilda_nn(:,ss)', 1);
		Y_nn = Y_nn + a;
	end
	Y(idces_nn) = Y_nn;
end

%%%%%%%%%% DGP 2 %%%%%%%%%%
%%%%%% Using MC integration
%%NumDraws = 100;
%%nu = normrnd(0,1,[NumDraws NumXhetero]); % NumDraws x NumXhetero
%%weights = 1/NumDraws * ones(NumDraws, 1); % NumDraws x 1
%%%%%% Using quadrature
%NumNodesPerDim = 20;
%[nu, weights] = GaussHermite_4_standard_MVN(NumXhetero, NumNodesPerDim); % [NumDraws x NumXhetero] and [NumDraws x 1]
%NumDraws = length(weights);
%%%%%%%%%%%%%%%%%%%%%%%
%
%Xhetero = X(:,idxes_heterog_coefs) .* reshape(nu', [1 NumXhetero NumDraws]); % Num_jm x NumXhetero x NumDraws
%Xhetero = permute(Xhetero, [1 3 2]); %  Num_jm x NumDraws x NumXhetero
%Xhetero = reshape(Xhetero, [Num_jm*NumDraws, NumXhetero]); % (Num_jm*NumDraws) x NumXhetero
%
%% From jm_2_mm_vec and NumDraws (given), construct jmd_2_md_vec
%Num_jm = size(jm_2_mm_vec, 1);
%jmd_2_mm_vec = reshape(repmat(jm_2_mm_vec, 1, NumDraws), [Num_jm*NumDraws 1]);
%jmd_2_dd_vec = reshape(repmat([1:NumDraws], Num_jm, 1), [Num_jm*NumDraws 1]);
%[~, ~, jmd_2_md_vec] = unique([jmd_2_dd_vec jmd_2_mm_vec], 'rows'); % (Num_jm*NumDraws) x 1
%
%V = X * beta_true + reshape(Xhetero * sigma_true, [Num_jm, NumDraws]); % Num_jm x NumDraws
%V = reshape(V, [Num_jm*NumDraws 1]); % (Num_jm*NumDraws) x 1
%
%Vmax = accumarray(jmd_2_md_vec, V, [NumMarkets*NumDraws 1], @max); % (NumMarkets*NumDraws) x 1
%V = V - Vmax(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
%tmp = log(accumarray(jmd_2_md_vec, exp(V), [NumMarkets*NumDraws 1])); % (NumMarkets*NumDraws) x 1
%log_pTilda = V - tmp(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
%p_Tilda = reshape(exp(log_pTilda), [Num_jm, NumDraws]); % Num_jm x NumDraws
%
%p = p_Tilda * weights; % Num_jm x 1
%
%Y = zeros(Num_jm, 1);
%for nn = 1:NumMarkets
%	idces_nn = find(jm_2_mm_vec == nn); % Jn x 1
%	p_nn = p(idces_nn); % Jn x 1
%	M_nn = M(nn); % scalar
%	Y(idces_nn) = mnrnd(M_nn, p_nn); % Jn x 1
%end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% ESTIMATE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define method and details for numerical integration
numinteg.method = 'GaussHermite';
numinteg.NumNodesPerDim = 20;
%numinteg.method = 'MonteCarlo';
%numinteg.NumDraws = 100;

% Launch estimation
[params_star, LL_star, LL_grad, FisherInfo, params_ses] = estimate_MMNL(M, X, Y, jm_2_mm_vec, idxes_heterog_coefs, numinteg);

disp(table(params_true, params_star, params_ses));
