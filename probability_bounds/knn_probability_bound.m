% KNN_PROBABILITY_BOUND robability bound for a knn classifier.
%
% This function provides a bound for
%
%   \max_i p(y_i | x_i, D)
%
% after adding additional points to the current training set, most
% useful with the optimal_search_bound_selector selection function.
%
% function bound = knn_probability_bound(responses, train_ind, test_ind, ...
%           weights, max_weights, pseudocount, num_positives)
%
% inputs:
%            data: an (n x d) matrix of input data
%       responses: an (n x 1) vector of responses (class 1 is
%                  tested against "any other class")
%       train_ind: an index into data/responses indicating the
%                  training points
%        test_ind: an index into data/responses indicating the test
%                  points
%         weights: an (n x n) matrix of weights
%     max_weights: precomputed max(weights)
%     pseudocount: a value in [0, 1] to use as a "pseudocount"
%   num_positives: the number of additional positive responeses to
%                  consider
%
% outputs:
%   bound: an upper bound for the probabilities of the test data
%
% copyright (c) roman garnett, 2011--2012

function bound = knn_probability_bound(~, train_ind, observed_labels, ...
          test_ind, num_positives, weights, max_weights, prior_alpha, ...
          prior_beta)

  % transform observed_labels to handle multi-class
  positive_ind = (observed_labels == 1);

  max_weight = max(max_weights(test_ind));

  successes = sum(weights(test_ind, train_ind( positive_ind)), 2);
  failures  = sum(weights(test_ind, train_ind(~positive_ind)), 2);

  max_alpha = (prior_alpha + successes + max_weight * num_positives);
  min_beta  = (prior_beta  + failures);

  bound = max(max_alpha ./ (max_alpha + min_beta));

end