rng(42);

% Set dimensions
N = 2000;
J = 400;
NumFirms = 20;

% Draw some random values for non-price utilities and price coefficients
V_nonprice = normrnd(0,1,[J N]); % J x N
beta_price = -2*abs(normrnd(0,1,[1 N])); % 1 x N

% Define size of consumer types
psi = ones(N,1); % N x 1

% Draw random values for firm marginal costs
marginal_costs = 10 + 5*rand(J,1);

%%% Define the ownership structure
% Draw a random firm for each product
ownership.product2firm = datasample(1:NumFirms, J)';
%ownership.product2firm = [1:J]';

% Make firm2products
ownership.firm2products = cell(NumFirms,1);
for ff = 1:NumFirms
	ownership.firm2products{ff} = find(ownership.product2firm == ff);
end


% Solve for optimal prices
tic
[price_eq, convergedFlag, NumIters, isEql] = solve_BLP_Bertrand_Nash_zetaFPI(marginal_costs, psi, V_nonprice, beta_price, ownership);
convergedFlag
NumIters
isEql

% Infer marginal costs based on optimal prices
marginal_costs2 = infer_marginal_costs(price_eq, psi, V_nonprice, beta_price, ownership);
toc

% Compare the two
disp(table(marginal_costs, marginal_costs2, price_eq));
max(abs(marginal_costs - marginal_costs2))
