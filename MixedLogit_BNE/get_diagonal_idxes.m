function [idxes] = get_diagonal_idxes(N)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Suppose I have a N x N matrix A and I want to extract the elements of the diagonal of A.
	% This function gives the indexes of these elements.
	% We can then do:
	%	idxes = get_diagonal_idxes(N);
	%	A(idxes) = values;
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Inputs:
	% N:			integer
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Outputs:
	% idxes:		array of integers
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	idxes = zeros(N, 1);
	for ii = 1:N
		idxes(ii) = (ii-1)*N + ii;
	end
end
