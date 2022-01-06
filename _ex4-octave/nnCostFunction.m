function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m,1) X];
a_1 = X;

z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);

a_2 = [ones(m,1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

h = a_3;
c1 = 1/m;
identity_matrix = eye(num_labels);
y_matrix = identity_matrix(y,:);
posClass = -(log(h) .* y_matrix);
negClass = log(1-h) .* (1-y_matrix);
doubleSum = sum(sum(posClass - negClass));

J = c1 * doubleSum;



% Part 2: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


c2 =  lambda/(2*m);
thetaReg1 = Theta1(:,2:end);
thetaReg2 = Theta2(:,2:end);
Reg = c2 * (sum(sum(thetaReg1.^2)) + sum(sum(thetaReg2.^2)));
J += Reg;


% Part 3: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = a_3 - y_matrix;
d2 = (d3 * thetaReg2) .* sigmoidGradient(z_2);

delta_2 = d3' * a_2;
delta_1 = d2' * a_1;

Theta1_grad = c1 * delta_1;
Theta2_grad = c1 * delta_2;

%regularization of gradients
gradReg1 = Theta1;
gradReg1(:,1) = 0;
gradReg1 = gradReg1 .* (lambda/m);

gradReg2 = Theta2;
gradReg2(:,1) = 0;
gradReg2 = gradReg2 .* (lambda/m);

Theta1_grad += gradReg1;
Theta2_grad += gradReg2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
