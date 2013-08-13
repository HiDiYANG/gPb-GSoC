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

void dsygv(double **A, double **B, int n, double *E);
void dsygv(double **A, double **B, int n, double *E, double **Evecs);

double *dsygv_ctof(double **in, int rows, int cols);
void dsygv_ftoc(double *in, double **out, int rows, int cols);
void dsygv_normalize(double **Evecs, int N);

extern "C" void dsygv_(int *itype, char *jobz, char *uplo, int *n,
		       double *a, int *lda, double *b, int *ldb,
		       double *w, double *work, int *lwork,
		       int *info);


void dsygv(double **A, double **B, int n, double *E)
{
  char jobz, uplo;
  int itype, lda, ldb, lwork, info, i;
  double *a, *b, *work, **Evecs;

  itype = 1; /* This sets the type of generalized eigenvalue problem
		that we are solving.  We have the possible values
		    1: Ax = lBx
		    2: ABx = lx
		    3: BAx = lx */

  jobz = 'N'; /* V/N indicates that eigenvectors should/should not
		 be calculated. */

  uplo = 'U'; /* U/L indicated that the upper/lower triangle of the
		 symmetric matrix is stored. */

  lda = n; // The leading dimension of the matrix A
  ldb = n; // The leading dimension of the matrix B
  
  lwork = 3*n-1;
  work = new double[lwork]; /* The work array to be used by dsygv and
			       its size. */

  a = dsygv_ctof(A, n, lda);
  b = dsygv_ctof(B, n, ldb); /* Here we convert the incoming arrays, assumed
			  to be in double index C form, to Fortran
			  style matrices. */

  dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, E, work, &lwork, &info);

  if ((info==0)&&(work[0]>lwork))
    cout << "The pre-set lwork value was sub-optimal for the job that\n"
	 << "you gave dsygv.  The used value was " << lwork
	 << " whereas " << work[0] << " is optimal.\n";

  delete a;
  delete b;
  delete work;
}


void dsygv(double **A, double **B, int n, double *E, double **Evecs)
{
  char jobz, uplo;
  int itype, lda, ldb, lwork, info, i;
  double *a, *b, *work;

  itype = 1;
  jobz = 'V';
  uplo = 'U';
  lda = n;
  ldb = n;
  
  lwork = 3*n-1;
  work = new double[lwork];

  a = dsygv_ctof(A, n, lda);
  b = dsygv_ctof(B, n, ldb);

  dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, E, work, &lwork, &info);

  if ((info==0)&&(work[0]>lwork))
    cout << "The pre-set lwork value was sub-optimal for the job that\n"
	 << "you gave dsygv.  The used value was " << lwork
	 << " whereas " << work[0] << " is optimal.\n";

  dsygv_ftoc(a, Evecs, n, lda);
  dsygv_normalize(Evecs, n);

  delete a;
  delete b;
  delete work;
}


double* dsygv_ctof(double **in, int rows, int cols)
{
  double *out;
  int i, j;

  out = new double[rows*cols];
  for (i=0; i<rows; i++) for (j=0; j<cols; j++) out[i+j*cols] = in[i][j];
  return(out);
}


void dsygv_ftoc(double *in, double **out, int rows, int cols)
{
  int i, j;

  for (i=0; i<rows; i++) for (j=0; j<cols; j++) out[i][j] = in[i+j*cols];
}


void dsygv_normalize(double **Evecs, int N)
{
  int i, j;
  double *norm;
  
  norm = new double[N];
  
  for (i=0; i<N; i++) {
    norm[i] = 0;
    for (j=0; j<N; j++) norm[i] += Evecs[j][i]*Evecs[j][i];
    norm[i] = sqrt(norm[i]);
  }
  
  for (i=0; i<N; i++) for (j=0; j<N; j++)
    Evecs[j][i] /= norm[i];
}
