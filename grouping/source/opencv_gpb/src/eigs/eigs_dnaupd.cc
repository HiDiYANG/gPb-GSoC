/*
  This is an example program using the header file
  dsaupd.h, my implementation of the ARPACK sparse matrix
  solver routine.  See that header file for more
  documentation on its use.
  
  For this example, we will work with the four by four
  matrix H[i,j] = i^j + j^i.  The output of this program
  should be:

  Eigenvalue  0: 0.151995
  Eigenvector 0: (0.610828, -0.72048, 0.324065, -0.0527271)
  Eigenvalue  1: 1.80498
  Eigenvector 1: (0.7614, 0.42123, -0.481884, 0.103072)
  Eigenvalue  2: 17.6352
  Eigenvector 2: (-0.216885, -0.547084, -0.764881, 0.26195)
  
  Scot Shaw
  7 September 1999
*/

// Begin with some standard include files.

#include <iostream>
#include <fstream>
#include "dnaupd.h"
using namespace std;

int main(int argc, char** argv)
{
  int n, nev, i, j;
  double **Evecs, *Evals;
  
  n = 4; // The order of the matrix

  tlen = n*n;
  T = new double*[tlen];
  for (i=0; i<tlen; i++) T[i] = new double[3];

  tlen = 0;
  for (i=0; i<n; i++) 
    for (j=0; j<n; j++) {
      T[tlen][0] = i;
      T[tlen][1] = j;
      T[tlen][2] = pow(i+1, j+1) + pow(j+1, i+1);
      tlen++;
    }

  nev = 2; // The number of values to calculate
  Evals = new double[nev];
  Evecs = new double*[n];
  
  for (size_t i=0; i<n; i++) 
    Evecs[i] = new double[nev];

  dnaupd(n, nev, Evals, Evecs);

  // Now we output the results.
  for (size_t i=0; i<nev; i++) {
    cout << "Eigenvalue  " << i << ": " << Evals[i] << endl;;
    cout << "Eigenvector " << i << ": (";
    for(size_t j=0; j<n; j++)
      cout << Evecs[j][i]<<", ";
    cout<<")"<<endl;
    }
}

void av(int n, double *in, double *out)
{
  for (size_t i=0; i<n; i++) 
    out[i] = 0;
  for (size_t i=0; i<tlen; i++) 
    out[(int)T[i][0]] += in[(int)T[i][1]] * T[i][2];
}
