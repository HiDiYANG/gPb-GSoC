echo on; clc;
%---------------------------------------------------------------------------
%A11_QR   MATLAB script file for implementing Algorithm 11.QR
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

clc; clear all; format long; delete output;

% - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
% HOUSEHOLDER`S  METHOD   followed by the   QR METHOD
%
%
% Remark. house2.m and qr2.m are used for Algorithm 11.QR

pause % Press any key to continue.

clc;

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
% Example 11.8, page 580.  Use Householder`s method of iteration
% to find the tridiagonal matrix  B = A    that is similar to  A.
%                                      n-2
%

A = [4    2    2    1;
     2   -3    1    5;
     2    1    3    1;
     1    5    1    2];

B = house2(A,1);

pause % Press any key to continue.

clc;
%............................................
% Begin section to print the results.
% Diary commands are included which write all
% the results to the Matlab textfile   output
%............................................
Ms1 = 'The matrix  A  is:';
Ms2 = 'The similar tridiagonal matrix B is:';
clc,disp(' '),disp(Ms1),disp(A),disp(Ms2),disp(B)
pause % Press any key to continue.

clc;
% - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
% Example  Use the QR method of iteration
% to reduce the tridiagonal matrix  B  to diagonal form.
%
epsilon = 5e-15;

D = qr2(B,epsilon,1);

pause % Press any key to continue.

clc;
%............................................
% Begin section to print the results.
% Diary commands are included which write all
% the results to the Matlab textfile   output
%............................................
Mx1 = 'Implementation of Householder reduction,';
Mx2 = 'followed by the QR method with shifts.';
Mx3 = 'The matrix  A  is:';
Mx4 = 'The similar diagonal matrix is:'
Mx5 = 'The eigenvalues are:';
clc,echo off,diary output,...
disp(' '),disp(Mx1),disp(Mx2),...
disp(' '),disp(Mx3),disp(A),disp(Mx4),...
disp(diag(D)),disp(Mx5),disp(D),diary off,echo on
