function A = house(A,show)
%---------------------------------------------------------------------------
%HOUSE    Householder`s method is used to reduce
%         a symmetric matrix to tridigonal form.
% Sample call
%   A = house(A,show)
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
% Algorithm 11.4 (Reduction to Tridiagonal Form).
% Section	11.4, Eigenvalues of Symmetric Matrices, Page 581
%---------------------------------------------------------------------------

if nargin==1, show = 0; end
[n,n] = size(A);
for k=1:(n-2),
  s = norm(A(k+1:n,k));                  % Start of Form_W
  if (A(k+1,k)<0), s = -s; end
  r = sqrt(2*(A(k+1,k) + s)*s);
  W(1:k) = zeros(1,k);                   % First K elements are 0
  W(k+1) = (A(k+1,k) + s)/r;  
  W(k+2:n) = A(k+2:n,k)'/r;              % End of Form_W
  V(1:k) = zeros(1,k); % Start of Form_V, First K elements are 0
  V(k+1:n) =  A(k+1:n,k+1:n)*W(k+1:n)';  % End of Form_V
  c = W(k+1:n)*V(k+1:n)';                % Start of Form_Q
  Q(1:k) = zeros(1,k);                   % First K elements are 0
  Q(k+1:n) = V(k+1:n) - c*W(k+1:n);      % End of Form_Q
	 A(k+2:n,k) = zeros(n-k-1,1);           % Start of Form_A
	 A(k,k+2:n) = zeros(1,n-k-1);
  A(k+1,k) = -s;
  A(k,k+1) = -s;
  A(k+1:n,k+1:n) = A(k+1:n,k+1:n) ...
                 - 2*W(k+1:n)'*Q(k+1:n) - 2*Q(k+1:n)'*W(k+1:n);  
                                         % End of Form_A
  if show==1,
    home; if k==1, clc; end; 
    Mx1 = '     Scalar s           Scalar r           Scalar c';
    Pts= [s r c];
    Vts= [W;V;Q]';
    Mx2 = '     Vector W           Vector V           Vector Q';
    disp(' '),disp(Mx1),disp(Pts),...
    disp(' '),disp(Mx2),disp(Vts),...
	   disp(' '),disp('Transformed matrix  A = '),disp(A)
    pause(1.5);
  end
end
