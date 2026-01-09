function [beta_star, LL_star, LL_grad, FisherInfo, beta_ses] = estimate_MMNL(M, X, Y, jm_2_mm_vec, jmd_2_md_vec, idxes_heterog_coefs, varargin)
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
	
	% TO DO: from jm_2_mm_vec and NumDraws (given), construct jmd_2_md_vec
	
	% Define objective function
	Xhomo = X; % Num_jm x NumXhomo
	[Num_jm,NumXhomo] = size(Xhomo);
	NumXhetero = length(idxes_heterog_coefs);
	NumDraws = size(weights,1);
	

	Xhetero = X(:,idxes_heterog_coefs) .* reshape(nu', [1 NumXhetero NumDraws]); % Num_jm x NumXhetero x NumDraws
	Xhetero = reshape(Xhetero, [Num_jm, NumXhetero, NumDraws]); % Num_jm x NumXhetero x NumDraws
	Xhetero = permute(Xhetero, [1 3 2]); % Num_jm x NumDraws x NumXhetero
	Xhetero = reshape(Xhetero, [Num_jm*NumDraws NumXhetero]); % (Num_jm*NumDraws) x NumXhetero
	
	
	M_rep = M(jm_2_mm_vec);
	LL_constant = sum(gammaln(M+1)) - sum(gammaln(Y+1));
	obj = @(params_val) loglikelihood_MMNL(Y, Xhomo, Xhetero, weights, M, M_rep, jmd_2_md_vec, params_val, LL_constant, true);
	
	
	% Define stating point function
	params0 = zeros(NumXhomo + NumXhetero, 1);
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for ii = 1:4
%	[beta_star, LL_star] = NelderMead(obj, params0); % TO DO
%	LL_grad = []; % TO DO
%	FisherInfo = []; % TO DO
%	beta_ses = []; % TO DO
%params0 = beta_star;
%end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	[beta_star, LL_star, LL_grad] = BFGS(obj, params0, 1e-6, 2000);	% TO DO
	FisherInfo = []; % TO DO
	beta_ses = []; % TO DO
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	[beta_star, LL_star, LL_grad, FisherInfo] = Newton_Raphson(obj, params0, 1e-6, 2000);	% TO DO
%	beta_ses = sqrt(diag(inv(FisherInfo))); % TO DO
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
