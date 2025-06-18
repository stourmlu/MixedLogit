function [mu, varargout] = demand(psi, V_nonprice, prices, beta_price, outputType, varargin)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% This function returns the expected demand mu + additional things useful in computing equilibrium.
	% Demand comes from a mixed logit model:
	%	V_ij = V_nonprice_ij + beta_price_i * price_j
	%   Prob_ij = exp(V_ij)/[1 + sum_k exp(V_ik)]
	%   mu_j = E[Y_j] = sum_i psi_i * Prob_ij
	% Each firm controls a set of products (a,b,...) among the J products and makes total profit:
	%	profit = sum_{j in {a,b,...}} mu_{a}*(price_j - MC_a)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% psi:						N x 1
	% V_nonprice:				J x N
	% prices:					J x 1
	% beta_price:				1 x N
	% varargin{1}=ownership:	object
	%	.firm2products:				cell(NumFirms,1)
	%		{ff}:						vector of product indexes between 1 and J that belong to firm ff
	%	.product2firm:				J x 1 (index of firm between 1 and NumFirms)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% mu:				J x 1
	% varargout
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% Compute probs
	probs = mixed_logit_probs(V_nonprice, prices, beta_price);
	
	% Compute mu
	mu = probs * psi; % J x 1
	
	if outputType == 1
		varargout{1} = probs;
		return;
	end
	
	% Compute psiBeta
	psiBeta = psi.*beta_price'; % N x 1

	% Get ownership structure
	ownership = varargin{1};
	NumFirms = length(ownership.firm2products);
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if outputType == 2		%%%%% Compute Omega %%%%%
		% Compute Omega_diag
		Omega_diag = (probs.^2 - probs) * psiBeta; % J x 1

		% Compute Omega_off_diag_blocks
		Omega_off_diag_blocks = cell(NumFirms,1);
		for bb = 1:NumFirms
			firmProducts = ownership.firm2products{bb};
			firmNumProducts = length(firmProducts);
			Omega_off_diag_blocks{bb} = zeros(firmNumProducts,firmNumProducts);
			for i1 = 2:firmNumProducts
				j1 = firmProducts(i1);
				probs1 = probs(j1,:);
				for i2 = 1:i1-1
					j2 = firmProducts(i2);
					probs2 = probs(j2,:);
					Omega_off_diag_blocks{bb}(i1,i2) = (probs1 .* probs2) * psiBeta;
					Omega_off_diag_blocks{bb}(i2,i1) = Omega_off_diag_blocks{bb}(i1,i2);
				end
			end
		end
		varargout{1} = Omega_diag; % J x 1
		varargout{2} = Omega_off_diag_blocks;
	end
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% Omega = Lambda + Gamma where Lambda is diagonal.
	%%%% This is useful when calling solve_BLP_Bertrand_Nash_zetaFPI to solve price equilibrium.
	if outputType == 3		%%%%% Compute Lambda_diag and Gamma %%%%%
		% Compute Lambda_diag
		Lambda_diag = -probs *psiBeta; % J x 1
		
		% Compute Gamma_diag
		Gamma_diag = probs.^2 *psiBeta; % J x 1
		
		% Compute Gamma_blocks
		Gamma_off_diag_blocks = cell(NumFirms,1);
		for bb = 1:NumFirms
			firmProducts = ownership.firm2products{bb};
			firmNumProducts = length(firmProducts);
			Gamma_off_diag_blocks{bb} = zeros(firmNumProducts,firmNumProducts);
			for i1 = 2:firmNumProducts
				j1 = firmProducts(i1);
				probs1 = probs(j1,:);
				for i2 = 1:i1-1
					j2 = firmProducts(i2);
					Gamma_off_diag_blocks{bb}(i1,i2) = (probs1 .* probs(j2,:)) * psiBeta;
					Gamma_off_diag_blocks{bb}(i2,i1) = Gamma_off_diag_blocks{bb}(i1,i2);
				end
			end
		end
		varargout{1} = Lambda_diag;
		varargout{2} = Gamma_diag;
		varargout{3} = Gamma_off_diag_blocks;
	end

end
