function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

pos = find(y == 1); neg = find(y == 0);
posX = X(pos, :);
negX = X(neg, :);
z1 = posX * theta;
g1 = sigmoid(z1);
z2 = negX * theta;
g2 = sigmoid(z2);
i1 = -1 * log(g1);
i2 = -1 * log(1 - g2);
J = (sum(i1) + sum(i2)) / m;

posMult = posX' * (g1 - 1);
negMult = negX' * g2;
grad = (posMult + negMult) / m;

% =============================================================

end
