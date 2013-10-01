/*
  In this header file, I have defined a simplified function
  call to the ARPACK solver routine for a simple, symmetric
  eigenvalue problem Av = lv.  The looping procedure and
  final extraction of eigenvalues and vectors is handled
  automatically.  Most of the parameters to the FORTRAN
  functions are hidden from the user, since most of them are
  determined from user input anyway.

  The remaining parameters to the function calls are as follows:

    dsaupd(int n, int nev, double *Evals)
    dsaupd(int n, int nev, doubel *Evals, double **Evecs)

    n: the order of the square matrix A
    nev: the number of eigenvalues to be found, starting at the
         bottom.  Note that the highest eigenvalues, or some
	 other choices, can be found.  For now, the choice of
	 the lowest nev eigenvalues is hard-coded.
    Evals: a one-dimensional array of length nev to hold the
           eigenvalues.
    Evecs: a two-dimensional array of size nev by n to hold the
           eigenvectors.  If this argument is not provided, the
	   eigenvectors are not calculated.  Note that the
	   vectors are stored as columns of Evecs, so that the
	   elements of vector i are the values Evecs[j][i].

  The function is overdefined, so that you can call it in either
  fashion and the appropriate version will be run.

  To use these function calls, there must be a function
  defined in the calling program of the form

    av(int n, double *in, double *out)

  where the function av finds out = A.in, and n is the order of
  the matrix A.  This function must be defined before the
  statement that includes this header file, as it needs to know
  about this function.  It is used in the looping procedure.

  Scot Shaw
  30 August 1999

  I changed the structure of av function.

  Di Yang
  29 August 2013
*/

using namespace std;

void av(double **T, int tlen, int n, double *in, double* out);

extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which,
                        int *nev, double *tol, double *resid, int *ncv,
                        double *v, int *ldv, int *iparam, int *ipntr,
                        double *workd, double *workl, int *lworkl,
                        int *info);

extern "C" void dseupd_(int *rvec, char *All, int *select, double *d,
                        double *v, int *ldv, double *sigma,
                        char *bmat, int *n, char *which, int *nev,
                        double *tol, double *resid, int *ncv, double *vv,
                        int *ldvv, int *iparam, int *ipntr, double *workd,
                        double *workl, int *lworkl, int *ierr);


void av(double** T, int tlen, int n, double *in, double *out)
{
    for(size_t i=0; i<n; i++)
        out[i] = 0;
    for(size_t i=0; i<tlen; i++)
        out[(int)T[i][0]] += in[(int)T[i][1]] * T[i][2];
}

void dsaupd(double** T, int tlen, int n, int nev, double *Evals, double **Evecs)
{
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "SM";
    double tol = 1e-3;
    double *resid = new double[n];
    int ncv = 4*nev;
    if (ncv>n) ncv = n;
    int ldv = n;
    double *v = new double[ldv*ncv];
    int *iparam = new int[11];
    iparam[0] = 1;
    iparam[2] = 3*n;
    iparam[6] = 1;
    int *ipntr = new int[11];
    double *workd = new double[3*n];
    double *workl = new double[ncv*(ncv+8)];
    int lworkl = ncv*(ncv+8);
    int info = 0;
    int rvec = 1;  // Changed from above
    int *select = new int[ncv];
    double *d = new double[2*ncv];
    double sigma;
    int ierr;
    char howmny[2] = "A";

    do {
        dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid,
                &ncv, v, &ldv, iparam, ipntr, workd, workl,
                &lworkl, &info);
        if ((ido==1)||(ido==-1))
            av(T, tlen, n, workd+ipntr[0]-1, workd+ipntr[1]-1);
    } while ((ido==1)||(ido==-1));

    if (info<0) {
        cout << "Error with dsaupd, info = " << info << "\n";
        cout << "Check documentation in dsaupd\n\n";
    } else {
        dseupd_(&rvec, howmny, select, d, v, &ldv, &sigma, bmat,
                &n, which, &nev, &tol, resid, &ncv, v, &ldv,
                iparam, ipntr, workd, workl, &lworkl, &ierr);

        if (ierr!=0) {
            cout << "Error with dseupd, info = " << ierr << "\n";
            cout << "Check the documentation of dseupd.\n\n";
        } else if (info==1) {
            cout << "Maximum number of iterations reached.\n\n";
        } else if (info==3) {
            cout << "No shifts could be applied during implicit\n";
            cout << "Arnoldi update, try increasing NCV.\n\n";
        }

        for (size_t i=0; i<nev; i++)
            Evals[i] = d[i];
        for (size_t i=0; i<nev; i++)
            for (size_t j=0; j<n; j++)
                Evecs[i][j] = v[i*n+j];

        delete resid;
        delete v;
        delete iparam;
        delete ipntr;
        delete workd;
        delete workl;
        delete select;
        delete d;
    }
}
