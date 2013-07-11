function A = house2(A,show)
%---------------------------------------------------------------------------
%HOUSE2   Householder`s method is used to reduce
%         a symmetric matrix to tridigonal form.
% Sample call
%   A = house2(A,show)
% Inputs
%   A   symmetric matrix
% Return
%   A   solution: tridiagonal matrix
%
% NUMERICAL METHODS: MATLAB Programs, (c) John H. Mathews 1995
% To accompany the text:
% NUMERICAL METHODS for Mathematics, Science and Engineering, 2nd Ed, 1992
% Prentice Hall, Englewood Cliffs, New Jersey, 07632, U.S.A.
% Prentice Hall, Inc.; USA, Canada, Mexico ISBN 0-13-624990-6
% Prentice Hall, International Editions:   ISBN 0-13-625047-5
% This free software is compliments of the author.
% E-mail address:      in%"mathews@fullerton.edu"
%
% Algorithm 11.QR (Householder Reduction to Tridiagonal Form).
%                 (Followed by the QR Method with Shifts).
% Section	11.4,  Eigenvalues of Symmetric Matrices, Page 574
%---------------------------------------------------------------------------

if nargin==1, show = 0; end
[n,n] = size(A);
for k = 2:n-1
  U = A(:,k-1);
  U(1:k-1) = zeros(k-1,1);
  U = U/norm(U);
  if U(k) < 0, U = -U; end
  U(k) = U(k) + 1;
  beta = -U(k);       % beta is  -(norm(U)^2)/2
  V = A*U/beta;
  gamma = V'*U/(2*beta);
  V = V + gamma*U;
  A  =  A + U*V' + V*U';
  A(k-1,k+1:n) = zeros(1,n-k);
  if show==1,
    home, clc
    disp(''),disp(['Householder similarity transformation No. ',...
    int2str(k-1)]),disp(''),disp(A),pause(1);
  end
end
