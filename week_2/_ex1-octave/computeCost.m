function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
%You need to return the following variables correctly
##modelPredictions = zeros(1,m);
##for i = 0:m,
##   modelPredictions(i) = theta' * X(i);
##end
modelPredictions = X * theta;
##sqrdErrors = zeros(1,m);
sqrdErrors = (modelPredictions - y).^ 2;
summation = sum(sqrdErrors);
J = 0.0;
J = summation/(2*m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% =========================================================================

end