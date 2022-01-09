function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta;
sqrdErrors = (h - y).^ 2;
summation = sum(sqrdErrors);
c1 = 1/(2*m);

thetaReg = theta(2:end);
Reg = lambda * sum((thetaReg .^ 2));

J = c1 * (summation + Reg);

c = 1/m;
err = h - y;
grad += c *(((X')*err)+(lambda*theta));
grad(1) -= c * (lambda*theta(1));


% =========================================================================

grad = grad(:);

end
