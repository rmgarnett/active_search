function scores = search_expected_utility(problem, train_ind, ...
          observed_labels, test_ind, model)

  probabilities = model(problem, train_ind, observed_labels, test_ind);

  % expected utility after adding x is simply u(D) + p(y = 1 | x, D)
  scores = search_utility(problem, train_ind, observed_labels) + ...
           probabilities(:, 1);

end