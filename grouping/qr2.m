function D = qr2(A,epsilon,show)
%---------------------------------------------------------------------------
%QR2   Householder reduction to tridiagonal form
%      followed by the QR method with shifts for
%      finding the eigenvalues of a symmetric tridigonal matrix.
% Sample call
%   D = qr2(A,epsilon,show)
% Inputs
%   A         symmetric matrix
%   epsilon   convergence tolerance
% Return
%   D         solution: vector of eigenvalues
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

if nargin==2, show = 0; end
[n,n] = size(A);
k = 1;
m = n;
cnt = 0;
while m > 1
  S = A(m-1:m,m-1:m);
  if abs(S(2,1)) < epsilon*(abs(S(1,1)) + abs(S(2,2)))
    A(m,m-1) = 0;
    A(m-1,m) = 0;
    m = m-1;
  else
    shift = eig(S);
    if abs(shift(1)-A(m,m))<abs(shift(2)-A(m,m))
      shift = shift(1); 
    else
      shift = shift(2); 
    end
    [Q,R] = qr(A-shift*eye(n));
    A = R*Q + shift*eye(n);
    cnt = cnt+1;
    if show==1,
      home,  if cnt==1, clc; end;
      disp(''),disp(['Symmetric tridiagonal QR iteration No. ',...
      int2str(cnt)]),disp(''),disp(A),pause(0.75);
    end
  end
  k = k+1;
end
D = diag(A);
