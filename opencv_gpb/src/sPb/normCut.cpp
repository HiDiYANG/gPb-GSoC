//
//    Normalise Cut with given affinity matrix (Sparse Mode)
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University.
//    and Willow Garage inc.
//    All rights reserved.
//
//

#include "normCut.h"
#include "dsaupd.h"
using namespace std;

namespace cv {
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
    for(size_t i = 0; i<tlen; i++) {
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
    cv::Mat ones = cv::Mat::ones(rows, cols, CV_32FC1);
    for (size_t i=1; i<nev; i++) {
        double max_p, min_p;
        for(size_t j=0; j<n; j++) {
            Evecs[i][j] = Evecs[i][j]/D[j];
            if(j == 0) {
                max_p = Evecs[i][j];
                min_p = Evecs[i][j];
            } else {
                if(max_p < Evecs[i][j])
                    max_p = Evecs[i][j];
                if(min_p > Evecs[i][j])
                    min_p = Evecs[i][j];
            }
        }
        sPb_raw[i-1] = cv::Mat(rows, cols, CV_64FC1, Evecs[i]);
        sPb_raw[i-1].convertTo(sPb_raw[i-1], CV_32FC1);
        cv::addWeighted(sPb_raw[i-1], 1.0, ones, -min_p, 0.0, sPb_raw[i-1]);
        cv::multiply(sPb_raw[i-1], ones, sPb_raw[i-1], 1/(max_p-min_p)/sqrt(Evals[i]));
    }
    //clean up
    delete[] Evecs;
    delete[] Evals;
    ones.release();
}

}
