function [probs] = mixed_logit_probs(V_nonprice, prices, beta_price)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% There are N types of individuals.
	% Each individual chooses, according to a multinomial logit model,
	% among J inside option and an outside option whose utility is zero.
	% The utility of inside option is: V_ij = V_nonprice_ij + beta_price_i*price_j.
	% The choice probs for the inside options are:
	% prob_ij = exp(V_ij) / [1 + exp_k V_ik]
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% V_nonprice:		J x N
	% prices:			J x 1
	% beta_price:		1 x N
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% probs:			J x N
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	V = V_nonprice + prices * beta_price; % J x N
	if false
		V0 = 0;
		eV = exp(V); % J x N
	else
		% Trick to avoid overflow
		V0 = min(-max(V,[],1), 100); % 1 x N
		eV = exp(V + V0); % J x N
	end
	sum_eV = exp(V0)+sum(eV, 1); % 1 x N
	probs = eV ./ sum_eV; % J x N
end
