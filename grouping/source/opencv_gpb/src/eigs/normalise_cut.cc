#include "normalise_cut.hh"
#include "dsaupd.h"
using namespace std;

void normalise_cut(double **T, //symmetric sparse matrix - Affinity Matrix
		   int tlen,   //the number of elements 
		   int rows,   //matrix order, also the length of diagnal matrix
		   int cols,
		   double *D,  //square root of Diagnoal matrix
		   int nev,    //The number of eigenvector desired
		   //outputs:
		   vector<cv::Mat> & sPb_raw)    
{
  double **Evecs, *Evals;
  int n = rows*cols;
  for(size_t i = 0; i<tlen; i++){
    if(T[i][0] == T[i][1])
      T[i][2] = (pow(D[int(T[i][0])], 2)-T[i][2])/D[int(T[i][0])]/D[int(T[i][1])];
    else
      T[i][2] = -T[i][2]/D[int(T[i][0])]/D[int(T[i][1])];
  }
  
  Evals = new double[nev];
  Evecs = new double*[nev];
  for (size_t i=0; i<nev; i++) 
    Evecs[i] = new double[n];

  dsaupd(T, tlen, n, nev, Evals, Evecs);

  sPb_raw.resize(nev-1);
  for (size_t i=1; i<nev; i++){
    for(size_t j=0; j<n; j++)
      Evecs[i][j] = Evecs[i][j]/D[j]/sqrt(Evals[i]);
    sPb_raw[i-1] = cv::Mat(cols, rows, CV_32FC1, Evecs[i]).t();
  }

  // Now we output the results.
  FILE* pFile;
  pFile = fopen("EigVects.txt", "w+");
  for (size_t i=0; i<nev; i++) {
    //cout << "Eigenvalue  " << i << ": " << Evals[i] << "\n";
    //cout << "Eigenvector " << i << ": ( ";
    for (size_t j=0; j<n; j++)
      fprintf(pFile, "%f ", Evecs[i][j]);
    fprintf(pFile, "\n");
    //cout<< Evecs[j][i] << " ";
    //cout<<")"<<endl;*/
  }
  fclose(pFile);

}
