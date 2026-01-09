function [LL, varargout] = loglikelihood_MMNL(Y, Xhomo, Xhetero, weights, M, M_rep, jmd_2_md_vec, params_val, LL_constant, returnNegative)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function computes the loglikelihood, its gradient and its Hessian for a Multinomial logit model.
	%
	% M_rep should be equal to M(jmd_2_md_vec). No check is done.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% Y:					Num_jm x 1
	% Xhomo:				Num_jm x NumXhomo
	% Xhetero:				(Num_jm*NumDraws) x NumXhetero
	% weights:				NumDraws x 1
	% M:					NumMarkets x 1
	% M_rep:				Num_jm x 1
	% jmd_2_md_vec:			Num_jm*NumDraws x 1 (integers between 1 and NumMarkets*NumDraws)
	% params_val:			(NumXhomo + NumXhetero) x 1
	% LL_constant:			scalar
	% returnNegative:		boolean
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% LL:            		scalar
	% grad:					L x 1
	% hessian:				L x L
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	%%% Read dimensions
	[Num_jm,NumXhomo]  = size(Xhomo);
	NumXhetero         = size(Xhetero,2);
	NumDraws           = size(weights,1);
	NumMarkets         = size(M,1);
	
	% Split params
	beta_val  = params_val(1:NumXhomo); % NumXhomo x 1
	sigma_val = params_val(NumXhomo+1:end); % NumXhetero x 1
	
	%%% Compute V
	V = Xhomo * beta_val + reshape(Xhetero * sigma_val, [Num_jm, NumDraws]); % Num_jm x NumDraws
	V = reshape(V, [Num_jm*NumDraws 1]); % (Num_jm*NumDraws) x 1
	Vmax = accumarray(jmd_2_md_vec, V, [NumMarkets*NumDraws 1], @max); % (NumMarkets*NumDraws) x 1
	V = V - Vmax(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
	tmp = log(accumarray(jmd_2_md_vec, exp(V), [NumMarkets*NumDraws 1])); % (NumMarkets*NumDraws) x 1
	log_pTilda = V - tmp(jmd_2_md_vec); % (Num_jm*NumDraws) x 1
	p_Tilda = reshape(exp(log_pTilda), [Num_jm, NumDraws]); % Num_jm x NumDraws
	
	logp = log(p_Tilda * weights); % Num_jm x 1
	
	%%% Compute LL
	LL = Y'*logp + LL_constant; % scalar
	if returnNegative
		LL = - LL;
	end
	
	if nargout <= 1; return; end;
	
	%%% Compute p
	p = exp(logp); % Num_jm x 1
	
	%%% Compute grad
	error('This code needs to be written properly.')
	tmp1 = weights' .* p_Tilda./p; % Num_jm x NumDraws
	tmp1 = reshape(tmp1, [Num_jm*NumDraws 1]); % (Num_jm*NumDraws) x 1
	tmp2 = zeros(Num_jm, NumXhomo); % Num_jm x NumXhomo
	p_Tilda = reshape(p_Tilda, [Num_jm*NumDraws 1]);
	for xx = 1:NumXhomo
		z1 = accumarray(jmd_2_md_vec, reshape(p_Tilda.*Xhomo(:,xx), [Num_jm*NumDraws 1]), [NumMarkets*NumDraws 1]); % (NumMarkets*NumDraws) x 1
		z2 = reshape(tmp1 .* z1(jmd_2_md_vec), [Num_jm NumDraws]); % Num_jm x NumDraws)
		tmp2(:,xx) =  sum(z2, 2); % Num_jm x 1
	end
	d_logp_dbeta = Xhomo - tmp2;
	
	d_logp_dsigma = xxx; % TO DO
	
	dLL_dbeta = Y' * d_logp_dbeta;  % 1 x NumXhomo
	dLL_sigma = Y' * d_logp_dsigma; % 1 x NumXhetero
	grad = [dLL_dbeta dLL_sigma]'; % (NumXhomo + NumXhetero) x 1
	
	if returnNegative
		grad = - grad;
	end
	varargout{1} = grad;
	if nargout <= 2; return; end;
	
	%%% Compute Hessian
	error('Need to compute hess');
	if returnNegative
		hess = - hess;
	end
	varargout{2} = hess;
end
