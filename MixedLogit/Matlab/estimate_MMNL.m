function [beta_star, LL_star, LL_grad, FisherInfo, beta_ses] = estimate_MMNL(M, X, Y, jm_2_mm_vec, idxes_heterog_coefs, varargin)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function estimates a multinomial logit model by maximum likelihood, using the Newton-Raphson algorithm.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% M:					N x 1        (number of tries for choice set with index nn)
	% X:					Num_jm x NumX  (covariates corresponding to each option, across all choice sets)
	% Y:					Num_jm x 1     (number of "successful tries" for that option in that choice set, across the M_n tries)
	% jm_2_mm_vec:			Num_jm x 1     (maps each option to the choice set it belongs to, values between 1 and NumMarkets)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% beta_star:           	NumX x 1
	% LL_star:				scalar
	% grad:					NumX x 1
	% FisherInfo:			NumX x NumX
	% beta_ses:				NumX x 1
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	if length(varargin) >=1;
		nu      = varargin{1};
		weights = varargin{2};
	else
		error('Need draw nu and weights'); % TO DO
	end
	
	% Read dimensions
	[Num_jm,NumXhomo] = size(X);
	NumXhetero = length(idxes_heterog_coefs);
	NumDraws = size(weights,1);
	
	% From jm_2_mm_vec and NumDraws (given), construct jmd_2_md_vec
	Num_jm = size(jm_2_mm_vec, 1);
	jmd_2_mm_vec = reshape(repmat(jm_2_mm_vec, 1, NumDraws), [Num_jm*NumDraws 1]);
	jmd_2_dd_vec = reshape(repmat([1:NumDraws], Num_jm, 1), [Num_jm*NumDraws 1]);
	[~, ~, jmd_2_md_vec] = unique([jmd_2_dd_vec jmd_2_mm_vec], 'rows'); % (Num_jm*NumDraws) x 1
	
	% From jm_2_mm_vec and NumDraws (given), construct jmd_2_jm_vec
	jmd_2_jm_vec = reshape(repmat([1:Num_jm], 1, NumDraws), [Num_jm*NumDraws 1]);

	% Define objective function
	Xhomo = X; % Num_jm x NumXhomo
	Xhetero = X(:,idxes_heterog_coefs) .* reshape(nu', [1 NumXhetero NumDraws]); % Num_jm x NumXhetero x NumDraws
	Xhetero = reshape(Xhetero, [Num_jm, NumXhetero, NumDraws]); % Num_jm x NumXhetero x NumDraws
	Xhetero = permute(Xhetero, [1 3 2]); % Num_jm x NumDraws x NumXhetero
	Xhetero = reshape(Xhetero, [Num_jm*NumDraws NumXhetero]); % (Num_jm*NumDraws) x NumXhetero
	
	
	M_rep = M(jm_2_mm_vec);
	LL_constant = sum(gammaln(M+1)) - sum(gammaln(Y+1));
	obj = @(params_val) loglikelihood_MMNL(Y, Xhomo, Xhetero, weights, M, M_rep, jmd_2_md_vec, jmd_2_jm_vec, params_val, LL_constant, true);
	
	
	% Define stating point function
%	params0 = zeros(NumXhomo + NumXhetero, 1);
	params0 = normrnd(0,1,[NumXhomo + NumXhetero, 1]);
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for ii = 1:4
%	[beta_star, LL_star] = NelderMead(obj, params0);
%	LL_grad = [];
%	FisherInfo = [];
%	beta_ses = [];
%params0 = beta_star;
%end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	[beta_star, LL_star, LL_grad] = BFGS(obj, params0, 1e-6, 2000, true);
	FisherInfo = [];
	beta_ses = [];
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	[beta_star, LL_star, LL_grad, FisherInfo] = Newton_Raphson(obj, params0, 1e-6, 2000);
%	beta_ses = sqrt(diag(inv(FisherInfo)));
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
