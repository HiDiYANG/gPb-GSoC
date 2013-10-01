/*
    Source code for computing ultrametric contour maps based on average boundary strength, as described in :

    P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik. From contours to regions: An empirical evaluation. In CVPR, 2009.

    Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
    March 2009.

    Modified to fit OpenCV implementation for GSoC2013
    By Di Yang
    di.yang@anu.edu.au
    August 2013
*/

#include "ucm_mean_pb.h"

using namespace std;

/*************************************************************/

/******************************************************************************/

#ifndef Order_node_h
#define Order_node_h

class Order_node
{
public:
    double energy;
    int region1;
    int region2;

    Order_node() {
        energy = 0.0;
        region1 = 0;
        region2 = 0;
    }

    Order_node( const double& e, const int& rregion1, const int& rregion2 )
    {
        energy = e;
        region1 = rregion1;
        region2 = rregion2;
    }

    ~Order_node() {}
    // LEXICOGRAPHIC ORDER on priority queue: (energy,label)
    bool operator < (const Order_node& x) const {
        return ( ( energy > x.energy ) ||(( energy == x.energy ) && (region1 > x.region1)) ||(( energy == x.energy ) && (region1 == x.region1)&& (region2 > x.region2)));
    }
};

#endif

/******************************************************************************/

#ifndef Neighbor_Region_h
#define Neighbor_Region_h

class Neighbor_Region
{
public:
    double energy;
    double total_pb;
    double bdry_length;

    Neighbor_Region()
    {
        energy = 0.0;
        total_pb = 0.0;
        bdry_length = 0.0;
    }

    Neighbor_Region(const Neighbor_Region& v)
    {
        energy = v.energy;
        total_pb = v.total_pb;
        bdry_length = v.bdry_length;
    }

    Neighbor_Region(const double& en, const double& tt, const double& bor )
    {
        energy = en;
        total_pb = tt;
        bdry_length = bor;
    }

    ~Neighbor_Region() {}

};

#endif

/******************************************************************************/

#ifndef Bdry_element_h
#define Bdry_element_h

class Bdry_element
{
public:
    int coord;
    int cc_neigh;

    Bdry_element() {}

    Bdry_element(const int& c, const int& v) {
        coord = c;
        cc_neigh = v;
    }

    Bdry_element(const Bdry_element& n) {
        coord = n.coord;
        cc_neigh = n.cc_neigh;
    }

    ~Bdry_element() {}

    bool operator ==(const Bdry_element& n) const {
        return ( ( coord == n.coord) && ( cc_neigh == n.cc_neigh) ) ;
    }
    // LEXICOGRAPHIC ORDER: (cc_neigh, coord)
    bool operator < (const Bdry_element& n) const {
        return ( (cc_neigh < n.cc_neigh) || ((cc_neigh == n.cc_neigh) && ( coord < n.coord)));
    }

};

#endif

/******************************************************************************/

#ifndef Region_h
#define Region_h

class Region
{
public:
    list<int> elements;
    map<int, Neighbor_Region, less<int> > neighbors;
    list<Bdry_element> boundary;

    Region() {}

    Region(const int& l) {
        elements.push_back(l);
    }

    ~Region() {}

    void merge( Region& r, int* labels, const int& label, double* ucm, const double& saliency, const int& son, const int& tx );

};

void Region::merge( Region& r, int* labels, const int& label, double* ucm, const double& saliency, const int& son, const int& tx )
{
    /* 			I. BOUNDARY        */

    // 	Ia. update father's boundary
    list<Bdry_element>::iterator itrb, itrb2;
    itrb = boundary.begin();
    while ( itrb != boundary.end() )
    {
        if( labels[(*itrb).cc_neigh] == son )
        {
            itrb2 = itrb;
            ++itrb;
            boundary.erase(itrb2);
        }
        else ++itrb;
    }

    //	Ib. move son's boundary to father
    for( itrb = r.boundary.begin(); itrb != r.boundary.end(); ++itrb )
    {
        if (ucm[(*itrb).coord] < saliency ) ucm[(*itrb).coord] = saliency;

        if ( labels[(*itrb).cc_neigh] != label )
            boundary.push_back( Bdry_element(*itrb) );

    }
    r.boundary.erase( r.boundary.begin(), r.boundary.end() );

    /* 			II. ELEMENTS      */

    for( list<int>::iterator p = r.elements.begin(); p != r.elements.end(); ++p ) labels[*p] = label;
    elements.insert( elements.begin(), r.elements.begin(), r.elements.end() );
    r.elements.erase( r.elements.begin(), r.elements.end() );


    /* 			III. NEIGHBORS        */

    map<int,Neighbor_Region, less<int> >::iterator itr, itr2;

    // 	IIIa. remove inactive neighbors from father
    itr = neighbors.begin();
    while( itr != neighbors.end() )
    {
        if ( labels[(*itr).first] != (*itr).first )
        {
            itr2 = itr;
            ++itr;
            neighbors.erase(itr2);
        } else ++itr;
    }

    // 	IIIb. remove inactive neighbors from son y and neighbors belonging to father
    itr = r.neighbors.begin();
    while ( itr != r.neighbors.end() )
    {
        if ( ( labels[(*itr).first] != (*itr).first ) || ( labels[(*itr).first] == label ) )
        {
            itr2 = itr;
            ++itr;
            r.neighbors.erase(itr2);
        } else ++itr;
    }
}

#endif

namespace cv
{
/*************************************************************/
void complete_contour_map(double* ucm, const int& txc, const int& tyc)
/* complete contour map by max strategy on Khalimsky space  */
{
    int vx[4] = { 1, 0, -1,  0 };
    int vy[4] = { 0, 1,  0, -1 };
    int nxp, nyp, cv;
    double maximo;

    for( int x = 0; x < txc; x = x + 2 ) for( int y = 0; y < tyc; y = y + 2 )
        {
            maximo = 0.0;
            for( int v = 0; v < 4; v++ )
            {
                nxp =  x + vx[v] ;
                nyp = y + vy[v];
                cv = nxp + nyp * txc;
                if ( (nyp >= 0) && (nyp < tyc) && (nxp < txc) && (nxp >= 0) && ( maximo < ucm[cv] ) )
                    maximo = ucm[cv];
            }
            ucm[x + y*txc] = maximo;
        }

}

/***************************************************************************************************************************/
void compute_ucm
(	double* local_boundaries, int* initial_partition, const int& totcc, double* ucm, const int& tx, const int& ty)
{
    // I. INITIATE
    int p,c;
    int* labels = new int[totcc];

    for(c = 0; c < totcc; c++ )
    {
        labels[c] = c;
    }

    // II. ULTRAMETRIC
    Region* R = new Region[totcc];
    priority_queue<Order_node, vector<Order_node>, less<Order_node> > merging_queue;
    double totalPb, totalBdry, dissimilarity;
    int v;

    for( p = 0; p < (2*tx+1)*(2*ty+1); p++ ) ucm[p] = 0.0;

    // INITIATE REGI0NS
    for ( c = 0; c < totcc; c++ ) R[c] = Region(c);

    // INITIATE UCM
    int vx[4] = { 1, 0, -1,  0};
    int vy[4] = { 0, 1,  0, -1};
    int nxp, nyp, cnp, xp, yp, label;

    for( p = 0; p < tx*ty; p++ )
    {
        xp = p%tx;
        yp = p/tx;
        for( v = 0; v < 4; v++ )
        {
            nxp =  xp + vx[v];
            nyp = yp + vy[v];
            cnp = nxp + nyp*tx;
            if ( (nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) && (initial_partition[cnp] != initial_partition[p]) )
                R[initial_partition[p]].boundary.push_back(Bdry_element(( xp + nxp + 1 ) + ( yp + nyp + 1 )*(2*tx+1), initial_partition[cnp]));
        }
    }

    // INITIATE merging_queue
    list<Bdry_element>::iterator itrb;
    for ( c = 0; c < totcc; c++ )
    {
        R[c].boundary.sort();

        label = (*R[c].boundary.begin()).cc_neigh;
        totalBdry = 0.0;
        totalPb = 0.0;

        for ( itrb = R[c].boundary.begin(); itrb != R[c].boundary.end(); ++itrb )
        {
            if ((*itrb).cc_neigh == label)
            {
                totalBdry++;
                totalPb += local_boundaries[(*itrb).coord];
            }
            else
            {
                R[c].neighbors[label] = Neighbor_Region(totalPb/totalBdry, totalPb, totalBdry);
                if( label > c )   merging_queue.push(Order_node(totalPb/totalBdry, c, label));
                label = (*itrb).cc_neigh;
                totalBdry = 1.0;
                totalPb = local_boundaries[(*itrb).coord];
            }

        }
        R[c].neighbors[label] = Neighbor_Region(totalPb/totalBdry, totalPb, totalBdry);
        if( label > c )   merging_queue.push(Order_node(totalPb/totalBdry, c, label));
    }


    //MERGING
    Order_node minor;
    int father, son;
    map<int,Neighbor_Region,less<int> >::iterator itr;
    double current_energy = 0.0;

    while ( !merging_queue.empty() )
    {
        minor = merging_queue.top();
        merging_queue.pop();
        if( (labels[minor.region1] == minor.region1) && (labels[minor.region2] == minor.region2)		&&
                (minor.energy == R[minor.region1].neighbors[minor.region2].energy) )
        {
            if (current_energy <= minor.energy) current_energy = minor.energy;
            else
            {
                printf("\n ERROR : \n");
                printf("\n current_energy = %f \n", current_energy);
                printf("\n minor.energy = %f \n\n", minor.energy);
                delete[] R;
                delete[] labels;
                cout<<" BUG: THIS IS NOT AN ULTRAMETRIC !!! "<<endl;;
            }

            dissimilarity =  R[minor.region1].neighbors[minor.region2].total_pb / R[minor.region1].neighbors[minor.region2].bdry_length ;

            if (minor.region1 < minor.region2)
            {
                son = minor.region1;
                father = minor.region2;
            }
            else
            {
                son = minor.region2;
                father = minor.region1;
            }

            R[father].merge(R[son], labels, father, ucm, dissimilarity, son, tx);

            // move and update neighbors
            while ( R[son].neighbors.size() > 0 )
            {
                itr = R[son].neighbors.begin();

                R[father].neighbors[(*itr).first].total_pb += (*itr).second.total_pb;
                R[(*itr).first].neighbors[father].total_pb += (*itr).second.total_pb;

                R[father].neighbors[(*itr).first].bdry_length += (*itr).second.bdry_length;
                R[(*itr).first].neighbors[father].bdry_length += (*itr).second.bdry_length;

                R[son].neighbors.erase(itr);
            }

            // update merging_queue
            for (itr = R[father].neighbors.begin(); itr != R[father].neighbors.end(); ++itr )
            {

                dissimilarity = R[father].neighbors[(*itr).first].total_pb / R[father].neighbors[(*itr).first].bdry_length;

                merging_queue.push(Order_node(dissimilarity, (*itr).first, father));
                R[father].neighbors[(*itr).first].energy = dissimilarity;
                R[(*itr).first].neighbors[father].energy = dissimilarity;

            }
        }
    }

    complete_contour_map(ucm, 2*tx+1, 2*ty+1 );

    delete[] R;
    delete[] labels;


}

/*************************************************************************************************/

void ucm_mean_pb(const cv::Mat & input1,
                 const cv::Mat & input2,
                 cv::Mat & output,
                 bool label)
{
    bool flag = label ? DOUBLE_SIZE : SINGLE_SIZE;
    double* local_boundaries = new double[input1.rows*input1.cols];
    double* ucm = new double[(input2.rows*2+1)*(input2.cols*2+1)];
    int* initial_partition = new int[input2.rows*input2.cols];
    int ind = 0;
    for(size_t j=0; j<input1.cols; j++)
        for(size_t i=0; i<input1.rows; i++) {
            local_boundaries[ind++] = double(input1.at<float>(i,j));
        }

    int totcc = -1;
    ind = 0;
    for(size_t j=0; j<input2.cols; j++)
        for(size_t i=0; i<input2.rows; i++) {
            initial_partition[ind] = input2.at<int>(i, j);
            if (totcc < initial_partition[ind])
                totcc = initial_partition[ind];
            ind++;
        }
    if (totcc < 0)
        cout<<"ERROR : number of connected components < 0" <<endl;
    totcc++;

    compute_ucm(local_boundaries, initial_partition, totcc, ucm, input2.rows, input2.cols);

    if(flag)
        output = cv::Mat(input2.cols*2+1, input2.rows*2+1, CV_64FC1, ucm).t();
    else {
        output = cv::Mat::zeros(input2.rows, input2.cols, CV_64FC1);
        cv::Mat ucm2 = cv::Mat(input2.cols*2+1, input2.rows*2+1, CV_64FC1, ucm).t();
        for(size_t i=0; i<input2.rows; i++)
            for(size_t j=0; j<input2.cols; j++)
                output.at<double>(i,j) = ucm2.at<double>(i*2,j*2);
    }
    output.convertTo(output, CV_32FC1);
    delete[] ucm;
    delete[] initial_partition;
    delete[] local_boundaries;
}
}
