function [firm_profits, varargout] = compute_firms_profit(price, marginal_costs, psi, V_nonprice, beta_price, ownership)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function returns the firm's profits in the following model:
	% Demand comes from a mixed logit model:
	%	V_ij = V_nonprice_ij + beta_price_i * price_j
	%   Prob_ij = exp(V_ij)/[1 + sum_k exp(V_ik)]
	%   mu_j = E[Y_j] = sum_i psi_i * Prob_ij
	% Each firm controls a set of products (a,b,...) among the J products and makes total profit:
	%	profit = sum_{j in {a,b,...}} mu_{a}*(price_j - MC_a)
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
	%	.product2firm:				J x 1 (index of firm between 1 and NumFirms)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% firm_profits:		NumFirms x 1
	% varargout{1}:		cell(NumFirms, 1)
	%	{oo}:				NumProducts_oo x 1  --> gives d_profit_oo / d_price_oo  (gradient of firm profit wrt to own prices)
	% varargout{2}:		cell(NumFirms, 1)
	%	{oo}:				NumProducts_oo x NumProducts_oo   --> gives d2profit_oo / d_price_oo d_price_oo'  (Hessian of firm profit wrt to own prices)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% Read dimensions
	NumFirms = length(ownership.firm2products);
	N = size(V_nonprice, 2);

	% Compute demand
	[mu, probs] = demand(psi, V_nonprice, price, beta_price, 1);
		% mu:     J x 1
		% probs: J x N

	% Compute markups
	markups = price-marginal_costs;
	
	% Unbundle ownership structure
	firm2products = ownership.firm2products;
	product2firm  = ownership.product2firm;

	% Compute firm profits (ignoring fixed costs)
	firm_profits = accumarray(product2firm, mu.*markups, [NumFirms 1]); % NumFirms x 1
	
	% Compute, for each firm, the first-order derivatives of the firm's profit wrt to the prices it charges
	if nargout >= 2
		psiBeta = psi.*beta_price'; % N x 1
		firm_gradients = cell(NumFirms, 1);
		
		dmu_k_dprice_j = cell(NumFirms, 1); % also useful for Hessian
		dmu_k_dprice_j = cell(NumFirms, 1); % also useful for Hessian
		
		dmu_j_dprice_j = (probs-probs.^2) * psiBeta; % J x 1
		dmu_j_dprice_j = dmu_j_dprice_j; % J x 1
		dprofit_dprice_singleProductFirms = mu + dmu_j_dprice_j.*markups; % J x 1
		
		for oo = 1:NumFirms
			products_oo = firm2products{oo};
			NumProducts_oo = length(products_oo);
			if NumProducts_oo == 1;
				dprofit_dprice_oo = dprofit_dprice_singleProductFirms(products_oo);
			else
				probs_oo = probs(products_oo,:); % NumProducts_oo x N
				psiBeta_probs_own_oo = probs_oo * psiBeta; % NumProducts_oo x 1
				psiBeta_probs_own_oo = diag(psiBeta_probs_own_oo); % NumProducts_oo x NumProducts_oo
				probs_cross_oo  = reshape(probs_oo, [NumProducts_oo 1 N]) .* reshape(probs_oo, [1 NumProducts_oo N]); % NumProducts_oo x NumProducts_oo x N
				psiBeta_probs_cross_oo = reshape(probs_cross_oo, [NumProducts_oo*NumProducts_oo N]) * psiBeta; % (NumProducts_oo*NumProducts_oo) x 1
				psiBeta_probs_cross_oo = reshape(psiBeta_probs_cross_oo, [NumProducts_oo NumProducts_oo]); % NumProducts_oo x NumProducts_oo
				dmu_k_dprice_j{oo} = psiBeta_probs_own_oo - psiBeta_probs_cross_oo; % % NumProducts_oo x NumProducts_oo
				dmu_k_dprice_j{oo} = dmu_k_dprice_j{oo};
				dprofit_dprice_oo = mu(products_oo) + dmu_k_dprice_j{oo}*markups(products_oo); % NumProducts_oo x 1
			end
			firm_gradients{oo} = dprofit_dprice_oo;
		end
		varargout{1} = firm_gradients;
	end
	
	
	% Compute, for each firm, the second-order derivatives of the firm's profit wrt to the prices it charges
	if nargout >= 3
		firm_Hessians = cell(NumFirms, 1);
		% Compute d2mu_j_dprice_j2 (second-order derivative)
		d2mu_j_dprice_j2 = (probs.*(1-probs).*(1-2*probs)) * (psiBeta.*beta_price'); % J x 1
	
		% Compute d2profit_j_dprice_j2
		d2profit_j_dprice_j2 = 2*dmu_j_dprice_j + d2mu_j_dprice_j2.*markups; % J x 1
		
		for oo = 1:NumFirms
			products_oo = firm2products{oo};
			NumProducts_oo = length(products_oo);
			if NumProducts_oo == 1;
				d2profit_dprice2_oo = d2profit_j_dprice_j2(products_oo); % NumProducts_oo x NumProducts_oo (= 1 x 1)
			else
				probs_oo = probs(products_oo,:); % NumProducts_oo x N
				B = eye(NumProducts_oo) - reshape(probs_oo, [NumProducts_oo 1 N]); % NumProducts_oo x NumProducts_oo x N
				C = reshape(B, [NumProducts_oo 1 NumProducts_oo N]) .* reshape(B, [1 NumProducts_oo NumProducts_oo N]); % NumProducts_oo x NumProducts_oo x NumProducts_oo x N
				D = reshape(probs_oo, [1 NumProducts_oo N]) .* B; % NumProducts_oo x NumProducts_oo x N
				E = C - reshape(D, [NumProducts_oo NumProducts_oo 1 N]); % NumProducts_oo x NumProducts_oo x NumProducts_oo x N
				F = reshape(probs_oo, [1 1 NumProducts_oo N]) .* E; % % NumProducts_oo x NumProducts_oo x NumProducts_oo x N
				d2mu_dprice_dprice = reshape(F, [NumProducts_oo*NumProducts_oo*NumProducts_oo N]) * (psiBeta .* beta_price'); % (NumProducts_oo*NumProducts_oo*NumProducts_oo) x 1
				d2mu_dprice_dprice = reshape(d2mu_dprice_dprice, [NumProducts_oo NumProducts_oo NumProducts_oo]); % NumProducts_oo x NumProducts_oo x NumProducts_oo
					% --> value in (j, j', k) gives d2mu_k/[dpricej_dpricej']
				d2mu_dprice_dprice = reshape(d2mu_dprice_dprice, [NumProducts_oo*NumProducts_oo NumProducts_oo]); % (NumProducts_oo*NumProducts_oo) x NumProducts_oo
				d2profit_dprice2_oo = 2*dmu_k_dprice_j{oo} + reshape(d2mu_dprice_dprice*markups(products_oo), [NumProducts_oo NumProducts_oo]); % NumProducts_oo x NumProducts_oo
				clear B C D E F;

			end
			firm_Hessians{oo} = d2profit_dprice2_oo;
		end
		varargout{2} = firm_Hessians;

	end
end
