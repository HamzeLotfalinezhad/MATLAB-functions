%% Optimal Data-Based Binning for Histograms
%%
% optBINS computes the optimal number of bins for a given one-dimensional
% data set. This optimization is based on the posterior probability for
% the number of bins
%
% Usage:
% optM = optBINS(data,minM,maxM);
%
% Where:
% data is a (1,N) vector of data points
% minM is the minimum number of bins to consider
% maxM is the maximum number of bins to consider
%
% This algorithm uses a brute-force search trying every possible bin number
% in the given range. This can of course be improved.
% Generalization to multidimensional data sets is straightforward.
%
% Created by Kevin H. Knuth on 17 April 2003
% Modified by Kevin H. Knuth on 15 March 2006
function optM = optBINS(data,minM,maxM)
if  size(data,1)>1
error('data dimensions must be (1,N)');
end
N = size(data,2);
% Simply loop through the different numbers of bins
% and compute the posterior probability for each.
logp = zeros(1,maxM);
for M = minM:maxM
n = hist(data,M); % Bin the data (equal width bins here)
part1 = N*log(M) + gammaln(M/2) - gammaln(N+M/2);
part2 = - M*gammaln(1/2) + sum(gammaln(n+0.5));
logp(M) = part1 + part2;
end
[maximum, optM] = max(logp);
return