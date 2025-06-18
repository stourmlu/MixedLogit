function [price_eq, convergedFlag, NumIters, isEql] = solve_BLP_Bertrand_Nash_zetaFPI(marginal_costs, psi, V_nonprice, beta_price, ownership, varargin)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function solves the Bertrand-Nash price equilibrium given the following demand:
	% 
	%	V_ij = V_nonprice_ij + beta_price_i * price_j
	%   Prob_ij = exp(V_ij)/[1 + sum_k exp(V_ik)]
	%   mu_j = E[Y_j] = sum_i psi_i * Prob_ij
	% The J products are "owned" by F firms that set prices so as to maximize their (expected) profit given this demand.
	% Firm f's pricing problem is:
	%	max_{p_j for j in J_f} sum_{j in J_f} (p_j - c_j) * mu_j
	%
	% To solve this problem, this function uses the zeta-FPI method developed by Morrow and Skerlos (2011).
	% This method seems robust.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% marginal_costs:			J x 1
	% psi:						N x 1
	% V_nonprice:				J x N
	% beta_price:				1 x N
	% ownership:				object
	%	.firm2products:				cell(NumFirms,1)
	%		{ff}:						vector of product indexes between 1 and J that belong to firm ff
	%	.product2firm:				J x 1 (index of firm between 1 and NumFirms)
	% price_start:				J x 1
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% price_eq:					J x 1
	% convergedFlag:			boolean
	% NumIters:					integer
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Get optional parameters: price_start, criterionStop and iterMax
	if length(varargin) >= 1
		price_start = varargin{1};
	else
		price_start = marginal_costs + 5e-1;
	end
	if length(varargin) >= 2
		criterionStop = varargin{2};
	else
		criterionStop = 1e-6;
	end
	if length(varargin) >= 3
		iterMax = varargin{3};
	else
		iterMax = 5000;
	end
	
	% Initialize markup
	markup_cur = price_start - marginal_costs;
	markup_prev = markup_cur + 1e2;
	NumIters = 0;
	
	% Iterate zeta steps
	while norm(markup_cur-markup_prev) > criterionStop && NumIters < iterMax
		markup_prev = markup_cur;
		price_cur = marginal_costs + markup_cur;
		[mu, Lambda_diag, Gamma_diag, Gamma_off_diag_blocks] = demand(psi, V_nonprice, price_cur, beta_price, 3, ownership);
		Gamma_markup = Gamma_diag.*markup_cur; % J x 1
		for bb = 1:length(ownership.firm2products)
			firmProducts = ownership.firm2products{bb};
			Gamma_markup(firmProducts) = Gamma_markup(firmProducts) + Gamma_off_diag_blocks{bb}*markup_cur(firmProducts);
		end
		markup_cur = (mu - Gamma_markup)./Lambda_diag;
		NumIters = NumIters + 1;
	end
	
	% Output equilibrium price and flag for convergenced reached
	price_eq = marginal_costs + markup_cur;
	if norm(markup_cur-markup_prev) <= criterionStop
		convergedFlag = true;
	else
		convergedFlag = false;
	end

	% Check if it is indeed an equilibrium
	isEql = check_price_equilibrium(price_eq, marginal_costs, psi, V_nonprice, beta_price, ownership);
end
