function [acc] = SKM(X1, X2, Y1, Y2, c, gamma)

% EigenTransfer Classification
% Input: (X1,Y1) - training data
%        (X2,Y2) - test data
%        c - the c value of SVM
%        gamma - regularization parameter of T
% Output: acc - classification accuracy

% implemented by LibSVM
m1 = size(X1,1);
m2 = size(X2,1);

K1 = calckernel('linear', kwidth, X1);
K2 = calckernel('linear', kwidth, X2);
K12 = calckernel('linear', kwidth, X2, X1);

K1 = full(K1);
K2 = full(K2);
K12 = full(K12);
K21 = K12';

% % non-regularized T
% A = K12 * pinv(K2) * K21;
% T = HalfInverse(K1) * Half(A);

% regularized T
% gamma = 10^(0);
A = K12 * pinv(K2) * K21;
T = HalfInverse(K1) * Half(A - 0.5 * gamma * pinv(K1));

H = [K1 K12];
G = H * H';
A = HalfInverse(G) * Half(K1);
U = H' * A;

H = [K21 K2];
G = H * H';
B = HalfInverse(G) * Half(K2);
V = H' * B;

K21 = V' * U * T;
K1 = T' * K1 * T;

n = size(K1,1);
K1 = [(1:n)', K1]; % include sample serial number as first column
opt = sprintf('-t 4 -c %d', c);
model = svmtrain(Y1, K1, opt);
n = size(K21,1);
K21 = [(1:n)', K21];
[pre_label, accuracy, dec_values] = svmpredict(Y2, K21, model);
acc = accuracy(1)/100;

function [Y] = HalfInverse(X)

% return Y = X^(-1/2)

[Ve Va] = eig(X);
va = diag(Va);
va(find(va<0)) = 0;
Tmp = va.^(0.5) + 10^(-5);
Y = diag(Tmp.^(-1));
Y = Ve * Y * Ve'; 


function [Y] = Half(X)

% return Y = X^(1/2)

[Ve Va] = eig(X);
va = diag(Va);
va(find(va<0)) = 0;
Tmp = va.^(0.5);
Y = Ve * diag(Tmp) * Ve'; 


