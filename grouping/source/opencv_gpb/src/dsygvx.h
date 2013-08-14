/*
  This file has my implementation of the LAPACK routine dsygv
  for C++.  This program solves for the eigenvalues and, if
  desired, the eigenvectors for a symmetric, generalized eigenvalue
  problem of the form Ax = lBx, where A and B are two symmetric
  matrices, x is an eigenvector, and l is an eigenvalue.  The
  program assumes that the upper triangles of the two matrices
  are stored.

  There are two function calls defined in this header, of the
  forms

    void dsygv(double **A, double **B, int n, double *E)
    void dsygv(double **A, double **B, int n, double *E, double **Evecs)

    A: the n by n matrix from the left hand side of the equation
    B: the n by n matrix from the right hand side of the equation
    n: the order of the square matrices A and B
    E: an n-element array to hold the eigenvalues l
    Evecs: an n by n matrix to hold the eigenvectors of the
           problem, if they are requested.

  The function call is defined twice, so that whether or not
  eigenvectors are called for the proper version is called.

  Scot Shaw
  28 September 1999
*/

#include <math.h>
using namespace std;

void dsygvx(double **A, double **B, int n, int il, int iu, double *E);
void dsygvx(double **A, double **B, int n, int il, int iu, double *E, 
	    double **Evecs);

double *dsygv_ctof(double **in, int rows, int cols);
void dsygv_ftoc(double *in, double **out, int rows, int cols);
void dsygv_normalize(double **Evecs, int rows, int cols);

extern "C" void dsygvx_(int *itype, 
			char *jobz,
			char *range,
			char *uplo, 
			int *n,
			double *a, 
			int *lda, 
			double *b,
			int *ldb,
			double *vl,
			double *vu,
			int *il,
			int *iu,
			double *abstol,
			int *m,
			double *w, 
			double *z,
			int *ldz,
			double *work, 
			int *lwork, 

			int *iwork,
			int *ifail, 
			int *info);


void dsygvx(double **A, double **B, int n, int il, int iu, double *E)
{
  char jobz, range, uplo;
  int itype, lda, ldb, m, ldz, lwork, info, *iwork, *ifail;
  double *a, *b, *z, *work, **Evecs;
  double abstol, vl=0, vu=0;

  itype = 1; /* This sets the type of generalized eigenvalue problem
		that we are solving.  We have the possible values
		    1: Ax = lBx
		    2: ABx = lx
		    3: BAx = lx */

  jobz = 'N'; /* V/N indicates that eigenvectors should/should not
		 be calculated. */

  range = 'I';
  
  uplo = 'U'; /* U/L indicated that the upper/lower triangle of the
		 symmetric matrix is stored. */

  lda = n; // The leading dimension of the matrix A
  ldb = n; // The leading dimension of the matrix B
  ldz = 1;
  
  abstol = 1e-4;
  lwork = 8*n;
  work = new double[lwork]; /* The work array to be used by dsygv and
			       its size. */
  iwork = new int[5*n];
  ifail = new int[n];
  
  a = dsygv_ctof(A, n, lda);
  b = dsygv_ctof(B, n, ldb); /* Here we convert the incoming arrays, assumed
			  to be in double index C form, to Fortran
			  style matrices. */

  dsygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, E, z, &ldz, work, &lwork, iwork, ifail, &info);

  if ((info==0)&&(work[0]>lwork))
    cout << "The pre-set lwork value was sub-optimal for the job that\n"
	 << "you gave dsygv.  The used value was " << lwork
	 << " whereas " << work[0] << " is optimal.\n";

  delete a;
  delete b;
  delete work;
  delete iwork;
}


void dsygvx(double **A, double **B, int n, int il, int iu, double *E, double **Evecs)
{
  char jobz, range, uplo;
  int itype, lda, ldb, m, ldz, lwork, info, *iwork, *ifail;
  double *a, *b, *z, *work;
  double abstol, vl=0, vu=0;

  itype = 1;
  jobz = 'V';
  range = 'I';
  uplo = 'U';
  lda = n;
  ldb = n;
  ldz = n;
  
  lwork =  8*n ;
  work = new double[lwork];
  iwork = new int[5*n];
  ifail = new int[n];
  z = new double[(iu-il+1)*ldz];

  a = dsygv_ctof(A, n, lda);
  b = dsygv_ctof(B, n, ldb);

  dsygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, E, z, &ldz, work, &lwork, iwork, ifail, &info);
  cout<<"here1"<<endl;
  
  if ((info==0)&&(work[0]>lwork))
    cout << "The pre-set lwork value was sub-optimal for the job that\n"
	 << "you gave dsygv.  The used value was " << lwork
	 << " whereas " << work[0] << " is optimal.\n";

  dsygv_ftoc(z, Evecs, ldz, iu-il+1);
  cout<<"here2"<<endl;
  dsygv_normalize(Evecs, ldz, iu-il+1);

  delete a;
  delete b;
  delete work;
  delete iwork;
}


double* dsygv_ctof(double **in, int rows, int cols)
{
  double *out;
  out = new double[rows*cols];
  for(size_t i=0; i<rows; i++) 
    for(size_t j=0; j<cols; j++) 
      out[i+j*rows] = in[i][j];
  return(out);
}


void dsygv_ftoc(double *in, double **out, int rows, int cols)
{
  for (size_t i=0; i<rows; i++) 
    for (size_t j=0; j<cols; j++) 
      out[i][j] = in[i+j*rows];
}


void dsygv_normalize(double **Evecs, int rows, int cols)
{
  int i, j;
  double *norm;
  
  norm = new double[cols];
  
  for (i=0; i<cols; i++) {
    norm[i] = 0;
    for (j=0; j<rows; j++) 
      norm[i] += Evecs[j][i]*Evecs[j][i];
    norm[i] = sqrt(norm[i]);
  }
  
  for (i=0; i<cols; i++) 
    for (j=0; j<rows; j++)
      Evecs[j][i] /= norm[i];
}
