/*
    Constrained segmentation by front propagation on Ultrametric Contour Map
    Source Code

    By Pablo Arbelaez
    arbelaez@eecs.berkeley.edu
    March 2008

    Modified to fit OpenCV implementation for GSoC 2013
    By Di Yang
    di.yang@anu.edu.au
    August 2013
*/

#include "uvt.h"

using namespace std;

#ifndef Active_h
#define Active_h

/*****************************************/
class Active {
public:
    double nrg;
    double lbl;
    int px;

    Active() {
        nrg = 0.0;
        lbl = -1;
        px = 0;
    }
    Active( const double& e, const double& l , const int& p) {
        nrg = e;
        lbl = l;
        px = p;
    }
    bool operator < ( const Active& x ) const {
        return ( (nrg > x.nrg) || ( (nrg == x.nrg) && ( lbl > x.lbl ) ) );
    }
};
#endif

namespace cv {

/*****************************************/
void  UVT( double *ucm, double* markers, const int& tx, const int& ty,
           double* labels, double* boundaries)
{
    //const double MAXDOUBLE = 1.7976931348623158e+308;
    // initialization
    priority_queue<Active, vector<Active>, less<Active> > band;
    bool* used = new bool[tx*ty];
    double* dist = new double[tx*ty];

    for (int p = 0; p < tx*ty; p++) {
        if( markers[p] > 0 ) {
            labels[p] = markers[p];
            dist[p] = 0.0;
            boundaries[p]=0.0;
            band.push( Active(dist[p], labels[p], p) );
        } else {
            labels[p] = -1;
            dist[p] = DBL_MAX;
        }
        used[p] = false;
    }

    // propagation
    int vx[4] = { 1,  0, -1,  0};
    int vy[4] = { 0,  1,  0, -1};
    int cp, nxp, nyp, cnp;
    double u;

    while ( !band.empty() ) {
        cp = band.top().px;
        band.pop();
        if (used[cp] == false) {
            for(int v = 0; v < 4; v++ ) {
                nxp = (cp%tx) + vx[v];
                nyp = (cp/tx) + vy[v];
                cnp = nxp + nyp*tx;
                if ( (nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) ) {
                    u = max(dist[cp], ucm[cnp]);
                    if(((u < dist[cnp])&&(labels[cnp]==-1)) ||
                            ((u == dist[cnp]) && (labels[cnp] < labels[cp]))) {
                        labels[cnp] = labels[cp];
                        dist[cnp] = u;
                        band.push( Active(dist[cnp],labels[cnp], cnp) );
                    }
                }
            }
            used[cp] = true;
        }
    }

    delete[] used;
    delete[] dist;
    for (int cp = 0; cp < tx*ty; cp++)
        for(int v = 0; v < 4; v++ ) {
            nxp = (cp%tx) + vx[v];
            nyp = (cp/tx) + vy[v];
            cnp = nxp + nyp*tx;
            if ( (nyp >= 0) && (nyp < ty) && (nxp < tx) &&
                    (nxp >= 0) && (labels[cnp] < labels[cp]))
                boundaries[cp]=1;
        }
}
/*****************************************/
/*           MEX INTERFACE               */
/*****************************************/

void uvt(const cv::Mat & ucm_mtr,
         const cv::Mat & seeds,
         cv::Mat & boundary,
         cv::Mat & labels,
         bool sz)
{
    bool flag = sz ? DOUBLE_SIZE : SINGLE_SIZE;
    int rows = ucm_mtr.rows;
    int cols = ucm_mtr.cols;
    double* ucm = new double[rows*cols];
    double* markers = new double[rows*cols];
    int ind = 0;
    for(size_t j=0; j<cols; j++)
        for(size_t i=0; i<rows; i++) {
            ucm[ind] = double(ucm_mtr.at<float>(i,j));
            markers[ind] = double(seeds.at<int>(i,j));
            ind++;
        }

    double* bdry = new double[rows*cols];
    double* lab = new double[rows*cols];

    UVT(ucm, markers, rows, cols, lab, bdry);

    if(flag) {
        boundary = cv::Mat(cols, rows, CV_64FC1, bdry).t();
        labels = cv::Mat(cols, rows, CV_64FC1, lab).t();
    } else {
        boundary = cv::Mat(int(rows/2), int(cols/2), CV_64FC1);
        labels = cv::Mat(int(rows/2), int(cols/2), CV_64FC1);
        cv::Mat temp1 = cv::Mat(cols, rows, CV_64FC1, bdry).t();
        cv::Mat temp2 = cv::Mat(cols, rows, CV_64FC1, lab).t();

        for(size_t i=0; i<int(rows/2); i++)
            for(size_t j=0; j<int(cols/2); j++) {
                boundary.at<double>(i, j) = temp1.at<double>(i*2,j*2);
                labels.at<double>(i, j) = temp2.at<double>(i*2,j*2);
            }
    }

    boundary.convertTo(boundary, CV_8UC1);
    labels.convertTo(labels, CV_8UC1);

    delete[] ucm;
    delete[] markers;
    delete[] bdry;
    delete[] lab;
}

void ucm2seg(const cv::Mat & ucm_mtr,
             cv::Mat & boundary,
             cv::Mat & labels,
             double thres,
             bool sz)
{
    bool flag = sz ? DOUBLE_SIZE : SINGLE_SIZE;

    if(flag) {
        boundary = cv::Mat(ucm_mtr.rows/2, ucm_mtr.cols/2, CV_8UC1);
        labels   = cv::Mat(ucm_mtr.rows/2, ucm_mtr.cols/2, CV_8UC1);
    } else {
        boundary = cv::Mat(ucm_mtr.rows, ucm_mtr.cols, CV_8UC1);
        labels   = cv::Mat(ucm_mtr.rows, ucm_mtr.cols, CV_8UC1);
    }
    cv::Mat_<uchar> bdry = cv::Mat_<uchar>(ucm_mtr.rows, ucm_mtr.cols);
    cv::Mat_<uchar> labs = cv::Mat_<uchar>(ucm_mtr.rows, ucm_mtr.cols);

    for(size_t i=0; i<ucm_mtr.rows; i++)
        for(size_t j=0; j<ucm_mtr.cols; j++) {
            if(ucm_mtr.at<float>(i,j) >= thres)
                bdry(i,j) = 255;
            else
                labs(i,j) = 1;
        }
    /*
    int index = 1;
    for(size_t i=0; i<labs.rows; i++)
      for(size_t j=0; j<labs.cols; j++){
    if(labs(i,j) == 1){
      cv::Point seed;
      seed.x = i;
      seed.y = j;
      cv::floodFill(labs, seed, cv::Scalar(index), 0, cv::Scalar(8));
      index++;
    }
      }
    */
    if(flag) {
        for(size_t i=0; i<boundary.rows; i++)
            for(size_t j=0; j<boundary.rows; j++) {
                boundary.at<uchar>(i,j) = bdry(i*2, j*2);
                labels.at<uchar>(i,j)   = labs(i*2, j*2);
            }
    } else {
        bdry.copyTo(boundary);
        labs.copyTo(labels);
    }
}
}
