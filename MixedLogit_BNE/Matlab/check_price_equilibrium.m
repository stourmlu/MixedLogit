function [isEql] = check_price_equilibrium(price, marginal_costs, psi, V_nonprice, beta_price, ownership)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Given a mixed logit model, this function tests whether the input price satisfies the necessary first-order and second-order conditions
	% of a Bertrand-Nash equilibrium.
	% Demand comes from a mixed logit model:
	%	V_ij = V_nonprice_ij + beta_price_i * price_j
	%   Prob_ij = exp(V_ij)/[1 + sum_k exp(V_ik)]
	%   mu_j = E[Y_j] = sum_i psi_i * Prob_ij
	% Each firm controls a set of products (a,b,...) among the J products and sets the prices to maximize its total profit:
	%	max_{price_a, price_b,...} sum_{j in {a,b,...}} mu_{a}*(price_j - MC_a)
	%
	% The function returns true if the price vector indeed satisfies the necessary FOC and SOC of a Bertrand-Nash equilibrium price, and false otherwise.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% price:				J x 1
	% marginal_costs:		J x 1
	% psi:					N x 1
	% V_nonprice:			J x N
	% beta_price:			1 x N
	% ownership:			object
	%	.firm2products:			cell(NumFirms,1)
	%		{ff}:					vector of product indexes between 1 and J that belong to firm ff
	%	.product2firm:			J x 1 (index of firm between 1 and NumFirms)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% isEql:				boolean
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	if any(isnan(price))
		disp('Price has NaN');
		isEql = false;
		return;
	end
	
	% Read dimensions
	J = size(price, 1);
		
	price0 = price;  % J x 1
	
	% Compute FOCs and SOCs (gradient and Hessian)
	[firm_profits0, firm_gradients, firm_Hessians] = compute_firms_profit(price0, marginal_costs, psi, V_nonprice, beta_price, ownership);
	
	% Initialize isEql to true
	isEql = true;

	% Check that profits are positive (they should be in the absence of fixed costs)
	if ~all(firm_profits0 >= 0)
		disp('Negative firm profit');
		min(firm_profits0)
		isEql = false;
		return;
	end;
	
	% Check FOCs
	for oo = 1:length(firm_gradients)
		if any(abs(firm_gradients{oo}) > 1e-5);
			disp('Gradient not zero');
			isEql = false;
		end;
	end
	
	% Check SOCs
	for oo = 1:length(firm_Hessians)
		eigenValues = eig(firm_Hessians{oo});
		if any(eigenValues >= 0);
			disp(sprintf('Hessian has nonnegative eigen values: firm %d did not reach a maximum.', oo));
			isEql = false;
			return;
		end;
	end
end
