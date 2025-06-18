function [marginal_costs, varargout] = infer_marginal_costs(price, psi, V_nonprice, beta_price, ownership)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function finds the marginal costs that rationalize observed prices, assuming they are the outcome of a 
	% Bertrand-Nash price equilibrium with a mixed logit model of demand:
	%	V_ij = V_nonprice_ij + beta_price_i * price_j
	%   Prob_ij = exp(V_ij)/[1 + sum_k exp(V_ik)]
	%   mu_j = E[Y_j] = sum_i psi_i * Prob_ij
	% The J products are "owned" by F firms that set prices so as to maximize their (expected) profit given this demand.
	% Firm f's pricing problem is:
	%	max_{p_j for j in J_f} sum_{j in J_f} (p_j - c_j) * mu_j
	%
	% This function takes as given:
	% - observed prices p_j (assumed to be an equilibrium outcome)
	% - demand model parameters (psi, V_nonprice, beta_price)
	% - ownership structure
	%
	% It returns the marginal costs faced by the firms for each product j (i.e. c_j).
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% price:				J x 1
	% psi:					N x 1
	% V_nonprice:			J x N
	% beta_price:			1 x N
	% ownership:			object
	%	.firm2products:			cell(NumFirms,1)
	%		{ff}:					vector of product indexes between 1 and J that belong to firm ff
	%	.product2firm:			J x 1 (index of firm between 1 and NumFirms)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% marginal_costs:		J x 1
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	[mu, Omega_diag, Omega_off_diag_blocks] = demand(psi, V_nonprice, price, beta_price, 2, ownership);
	markups = mu./Omega_diag; % J x 1
	
	% Adjust markups for collusion, taking into account Omega_off_diag_blocks
	for bb = 1:length(Omega_off_diag_blocks)
		firmProducts = ownership.firm2products{bb};
		Omega_block = Omega_off_diag_blocks{bb};
		Omega_block = setDiagonal(Omega_block, Omega_diag(firmProducts));
		markups(firmProducts) = Omega_block\mu(firmProducts);
	end
	marginal_costs = price - markups; % J x 1
	
	if nargout >= 1
		varargout{1} = mu;
	end
end
