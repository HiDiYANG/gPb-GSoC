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

#include "uvt.hh"

using namespace std;

#ifndef Active_h
#define Active_h

/*****************************************/
class Active
{
    public:
        double nrg;
        double lbl;
        int px;
        
        
        Active() { nrg = 0.0; lbl = -1; px = 0;}
        
        Active( const double& e, const double& l , const int& p) { nrg = e; lbl = l; px = p;}
        
        bool operator < ( const Active& x ) const { return ( (nrg > x.nrg) || ( (nrg == x.nrg) && ( lbl > x.lbl ) ) ); } 
        
};
#endif


namespace cv{

/*****************************************/
void  UVT( double *ucm, double* markers, const int& tx, const int& ty,
double* labels, double* boundaries)
{
    //const double MAXDOUBLE = 1.7976931348623158e+308;
    // initialization
    priority_queue<Active, vector<Active>, less<Active> > band;
    bool* used = new bool[tx*ty];
    double* dist = new double[tx*ty]; 
		
	for (int p = 0; p < tx*ty; p++)
    {
        if( markers[p] > 0 )
        {
            labels[p] = markers[p];
            dist[p] = 0.0;
			boundaries[p]=0.0;
            band.push( Active(dist[p], labels[p], p) );
        }
        else
        {
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
    
    while ( !band.empty() )
    {
        cp = band.top().px; band.pop();
        if (used[cp] == false)
        {
            for(int v = 0; v < 4; v++ )
            {
                nxp = (cp%tx) + vx[v]; nyp = (cp/tx) + vy[v]; cnp = nxp + nyp*tx;
                if ( (nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) )
                {
 						u = max(dist[cp], ucm[cnp]);

					if (((u < dist[cnp])&&(labels[cnp]==-1)) || ( (u == dist[cnp]) && (labels[cnp] < labels[cp] ) ) )
                    {
                        labels[cnp] = labels[cp];
                        
						dist[cnp] = u;

                        band.push( Active(dist[cnp],labels[cnp], cnp) );
                    }
                }
            }
            used[cp] = true;
        }
    }
    
    delete[] used; delete[] dist;
	
	for (int cp = 0; cp < tx*ty; cp++)
		for(int v = 0; v < 4; v++ )
        {
			nxp = (cp%tx) + vx[v]; nyp = (cp/tx) + vy[v]; cnp = nxp + nyp*tx;
  			if ( (nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) && (labels[cnp] < labels[cp]))
				boundaries[cp]=1;
		}


}
/*****************************************/
/*           MEX INTERFACE               */
/*****************************************/

void uvt(const cv::Mat & ucm_mtr,
	 const cv::Mat & seeds,
	 cv::Mat & boundary,
	 cv::Mat & labels)
{ 
  int rows = ucm_mtr.rows;
  int cols = ucm_mtr.cols;
  double* ucm = new double[rows*cols];
  double* markers = new double[rows*cols];
  int ind = 0;
  for(size_t j=0; j<cols; j++)
    for(size_t i=0; i<rows; i++){
      ucm[ind] = double(ucm_mtr.at<float>(i,j));
      markers[ind] = double(seeds.at<float>(i,j));
      ind++;
    }

  double* bdry = new double[rows*cols];
  double* lab = new double[rows*cols];
  
  UVT(ucm, markers, rows, cols, lab, bdry);
  
  boundary = cv::Mat(cols, rows, CV_64FC1, bdry).t();
  labels = cv::Mat(cols, rows, CV_64FC1, lab).t();
  
}
}
