//
//    globalPb
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University.
//    and Willow Garage inc.
//    All rights reserved.
//

#include "filters.h"
#include "globalPb.h"
#include "buildW.h"
#include "normCut.h"

using namespace std;

namespace
{ static double*
_gPb_Weights(int nChannels)
{
    double *weights = new double[13];
    if(nChannels == 3) {
        weights[0] = 0.0;
        weights[1] = 0.0;
        weights[2] = 0.0039;
        weights[3] = 0.0050;
        weights[4] = 0.0058;
        weights[5] = 0.0069;
        weights[6] = 0.0040;
        weights[7] = 0.0044;
        weights[8] = 0.0049;
        weights[9] = 0.0024;
        weights[10]= 0.0027;
        weights[11]= 0.0170;
        weights[12]= 0.0094;
    } else {
        weights[0] = 0.0;
        weights[1] = 0.0;
        weights[2] = 0.0054;
        weights[3] = 0.0;
        weights[4] = 0.0;
        weights[5] = 0.0;
        weights[6] = 0.0;
        weights[7] = 0.0;
        weights[8] = 0.0;
        weights[9] = 0.0048;
        weights[10]= 0.0049;
        weights[11]= 0.0264;
        weights[12]= 0.0090;
    }
    return weights;
}

static double*
_mPb_Weights(int nChannels)
{
    double *weights = new double[12];
    if(nChannels == 3) {
        weights[0] = 0.0146;
        weights[1] = 0.0145;
        weights[2] = 0.0163;
        weights[3] = 0.0210;
        weights[4] = 0.0243;
        weights[5] = 0.0287;
        weights[6] = 0.0166;
        weights[7] = 0.0185;
        weights[8] = 0.0204;
        weights[9] = 0.0101;
        weights[10]= 0.0111;
        weights[11]= 0.0141;
    } else {
        weights[0] = 0.0245;
        weights[1] = 0.0220;
        weights[2] = 0.0;
        weights[3] = 0.0;
        weights[4] = 0.0;
        weights[5] = 0.0;
        weights[6] = 0.0;
        weights[7] = 0.0;
        weights[8] = 0.0;
        weights[9] = 0.0208;
        weights[10]= 0.0210;
        weights[11]= 0.0229;
    }
    return weights;
}
}

namespace cv
{
void
pb_parts_final_selected(vector<cv::Mat> & layers,
                        vector<vector<cv::Mat> > & gradients)
{
    int n_ori  = 8;                           // number of orientations
    int length = 7;
    double bg_smooth_sigma = 0.07;             // bg histogram smoothing sigma
    double cg_smooth_sigma = 0.04;            // cg histogram smoothing sigma
    double sigma_tg_filt_sm = 2.0;            // sigma for small tg filters
    double sigma_tg_filt_lg = sqrt(2.0)*2.0;  // sigma for large tg filters

    int bins[2] = {25, 64};
    int radii[4] = {3, 5, 10, 20};

    vector<cv::Mat> filters;

    filters.resize(3);
    cv::Mat color, grey, ones;
    cv::merge(layers, color);
    cv::cvtColor(color, grey, CV_BGR2GRAY);
    ones = cv::Mat::ones(color.rows, color.cols, CV_32FC1);

    // Histogram filter generation
    cv::gaussianFilter1D(double(bins[0])*bg_smooth_sigma, 0, false, filters[0]);
    cv::gaussianFilter1D(double(bins[0])*cg_smooth_sigma, 0, false, filters[1]);
    cv::transpose(filters[0], filters[0]);
    cv::transpose(filters[1], filters[1]);
    // impluse filter
    filters[2] = cv::Mat::zeros(1, length, CV_32FC1);
    filters[2].at<float>(0, (length-1)/2) = 1.0;

    // Normalize color channels
    color.convertTo(color, CV_32FC3);
    cv::split(color, layers);
    for(size_t c=0; c<3; c++)
        cv::multiply(layers[c], ones, layers[c], 1.0/255.0);
    cv::merge(layers, color);

    // Color convert, including gamma correction
    cv::cvtColor(color, color, CV_BGR2Lab);

    // Normalize Lab channels
    cv::split(color, layers);
    for(size_t c=0; c<3; c++)
        for(size_t i=0; i<layers[c].rows; i++) {
            for(size_t j=0; j<layers[c].cols; j++) {
                if(c==0)
                    layers[c].at<float>(i,j) = layers[c].at<float>(i,j)/100.0;
                else
                    layers[c].at<float>(i,j) = (layers[c].at<float>(i,j)+73.0)/168.0;
                if(layers[c].at<float>(i,j) < 0.0)
                    layers[c].at<float>(i,j) = 0.0;
                else if(layers[c].at<float>(i,j) > 1.0)
                    layers[c].at<float>(i,j) = 1.0;

                //quantize color channels
                float bin = floor(layers[c].at<float>(i,j)*float(bins[0]));
                if(bin == float(bins[0])) bin--;
                layers[c].at<float>(i,j)=bin;
            }
        }
    layers.resize(4);

    /********* END OF FILTERS INTIALIZATION ***************/
    cout<<" --->  computing texton ... "<<endl;
    cv::textonRun(grey, layers[3], n_ori, bins[1], sigma_tg_filt_sm, sigma_tg_filt_lg);
    cout<<" --->  computing bg cga cgb tg ... "<<endl;
    gradients.resize(layers.size()*3);
    //parallel_for_gradients(layers, filters, gradients, n_ori, bins, radii);

    for(size_t i=0; i<gradients.size(); i++)
        cv::gradient_hist_2D(layers[i/3], radii[i-((i/3)*3-int(i>2))], n_ori,
                             bins[i/9], filters[i/3-int(i>5)], gradients[i]);
}

void
nonmax_oriented_2D(const cv::Mat & mPb_max,
                   const cv::Mat & index,
                   cv::Mat & output,
                   double o_tol)
{
    int rows = mPb_max.rows;
    int cols = mPb_max.cols;
    //mPb_max.copyTo(output);
    output=cv::Mat::zeros(rows, cols, CV_32FC1);
    for(size_t i=0; i<rows; i++)
        for(size_t j=0; j<cols; j++) {
            double ori = index.at<float>(i,j);
            double theta = ori; //ori+M_PI/2.0;
            theta -= double(int(theta/M_PI))*M_PI;
            double v = mPb_max.at<float>(i,j);
            int ind0a_x = 0, ind0a_y = 0, ind0b_x = 0, ind0b_y = 0;
            int ind1a_x = 0, ind1a_y = 0, ind1b_x = 0, ind1b_y = 0;
            double d = 0;
            bool valid0 = false, valid1 = false;
            double theta2 = 0;

            if(theta < 1e-6)
                theta = 0.0;

            if(theta == 0) {
                valid0 = (i>0);
                valid1 = (i<(rows-1));
                if(valid0) {
                    ind0a_x = i-1;
                    ind0a_y = j;
                    ind0b_x = i-1;
                    ind0b_y = j;
                }
                if(valid1) {
                    ind1a_x = i+1;
                    ind1a_y = j;
                    ind1b_x = i+1;
                    ind1b_y = j;
                }
            } else if(theta < M_PI/4.0) {
                d = tan(theta);
                theta2 = theta;
                valid0 = ((i>0) && (j>0));
                valid1 = (i<(rows-1) && j<(cols-1));
                if(valid0) {
                    ind0a_x = i-1;
                    ind0a_y = j;
                    ind0b_x = i-1;
                    ind0b_y = j-1;
                }
                if(valid1) {
                    ind1a_x = i+1;
                    ind1a_y = j;
                    ind1b_x = i+1;
                    ind1b_y = j+1;
                }
            } else if(theta < M_PI/2.0) {
                d = tan(M_PI/2.0 - theta);
                theta2 = M_PI/2.0 - theta;
                valid0 = ((i>0) && (j>0));
                valid1 = (i<(rows-1) && j<(cols-1));
                if(valid0) {
                    ind0a_x = i;
                    ind0a_y = j-1;
                    ind0b_x = i-1;
                    ind0b_y = j-1;
                }
                if(valid1) {
                    ind1a_x = i;
                    ind1a_y = j+1;
                    ind1b_x = i+1;
                    ind1b_y = j+1;
                }
            } else if(theta == M_PI/2.0) {
                valid0 = (j>0);
                valid1 = (j<(cols-1));
                if(valid0) {
                    ind0a_x = i;
                    ind0a_y = j-1;
                    ind0b_x = i;
                    ind0b_y = j-1;
                }
                if(valid1) {
                    ind1a_x = i;
                    ind1a_y = j+1;
                    ind1b_x = i;
                    ind1b_y = j+1;
                }
            } else if(theta < 3.0*M_PI/4.0) {
                d = tan(theta - M_PI/2.0);
                theta2 = theta - M_PI/2.0;
                valid0 = ((i<rows-1) && (j>0));
                valid1 = (i>0 && j<(cols-1));
                if(valid0) {
                    ind0a_x = i;
                    ind0a_y = j-1;
                    ind0b_x = i+1;
                    ind0b_y = j-1;
                }
                if(valid1) {
                    ind1a_x = i;
                    ind1a_y = j+1;
                    ind1b_x = i-1;
                    ind1b_y = j+1;
                }
            } else {
                d = tan(M_PI-theta);
                theta2 = M_PI-theta;
                valid0 = ((i<rows-1) && (j>0));
                valid1 = (i>0 && j<(cols-1));
                if(valid0) {
                    ind0a_x = i+1;
                    ind0a_y = j;
                    ind0b_x = i+1;
                    ind0b_y = j-1;
                }
                if(valid1) {
                    ind1a_x = i-1;
                    ind1a_y = j;
                    ind1b_x = i-1;
                    ind1b_y = j+1;
                }
            }

            if(d > 1.0 || d < 0.0)
                cout<<"something wrong"<<endl;

            if(valid0 && valid1) {
                double v0a = 0, v0b = 0, v1a = 0, v1b = 0;
                double ori0a = 0, ori0b = 0, ori1a = 0, ori1b = 0;
                if(valid0) {
                    v0a = mPb_max.at<float>(ind0a_x, ind0a_y);
                    v0b = mPb_max.at<float>(ind0b_x, ind0b_y);
                    ori0a = index.at<float>(ind0a_x, ind0a_y)-ori;
                    ori0b = index.at<float>(ind0b_x, ind0b_y)-ori;
                }
                if(valid1) {
                    v1a = mPb_max.at<float>(ind1a_x, ind1a_y);
                    v1b = mPb_max.at<float>(ind1b_x, ind1b_y);
                    ori1a = index.at<float>(ind1a_x, ind1a_y)-ori;
                    ori1b = index.at<float>(ind1b_x, ind1b_y)-ori;
                }
                ori0a -= double(int(ori0a/(2*M_PI))) * (2*M_PI);
                ori0b -= double(int(ori0b/(2*M_PI))) * (2*M_PI);
                ori1a -= double(int(ori1a/(2*M_PI))) * (2*M_PI);
                ori1b -= double(int(ori1b/(2*M_PI))) * (2*M_PI);
                if(ori0a >= M_PI) {
                    ori0a = 2*M_PI - ori0a;
                }
                if(ori0b >= M_PI) {
                    ori0b = 2*M_PI - ori0b;
                }
                if(ori1a >= M_PI) {
                    ori1a = 2*M_PI - ori1a;
                }
                if(ori1b >= M_PI) {
                    ori1b = 2*M_PI - ori1b;
                }
                if(ori0a >= M_PI/2.0) {
                    ori0a = M_PI - ori0a;
                }
                if(ori0b >= M_PI/2.0) {
                    ori0b = M_PI - ori0b;
                }
                if(ori1a >= M_PI/2.0) {
                    ori1a = M_PI - ori1a;
                }
                if(ori1b >= M_PI/2.0) {
                    ori1b = M_PI - ori1b;
                }

                ori0a = (ori0a <= o_tol) ? 0.0 : (ori0a - o_tol);
                ori0b = (ori0b <= o_tol) ? 0.0 : (ori0b - o_tol);
                ori1a = (ori1a <= o_tol) ? 0.0 : (ori1a - o_tol);
                ori1b = (ori1b <= o_tol) ? 0.0 : (ori1b - o_tol);

                double v0 = (1.0-d)*v0a*cos(ori0a) + d*v0b*cos(ori0b);
                double v1 = (1.0-d)*v1a*cos(ori1a) + d*v1b*cos(ori1b);
                if((v>v0) && (v>v1)) {
                    v = 1.2*v;
                    if(v > 1.0) v = 1.0;
                    if(v < 0.0) v = 0.0;
                    output.at<float>(i, j) = v;
                }
            }
        }
}

void
MakeFilter(const int radii,
           const double theta,
           cv::Mat & kernel)
{
    double ra, rb, ira2, irb2;
    double sint, cost, ai, bi;
    double x[5] = {0};
    int wr;
    cv::Mat A = cv::Mat::zeros(3, 3, CV_32FC1);
    cv::Mat y = cv::Mat::zeros(3, 1, CV_32FC1);
    ra = MAX(1.5, double(radii));
    rb = MAX(1.5, double(radii)/4);
    ira2 = 1.0/(pow(ra, 2));
    irb2 = 1.0/(pow(rb, 2));
    wr = int(MAX(ra, rb));
    kernel = cv::Mat::zeros(2*wr+1, 2*wr+1, CV_32FC1);

    sint = sin(theta);
    cost = cos(theta);
    for(size_t i = 0; i <= 2*wr; i++)
        for(size_t j = 0; j <= 2*wr; j++) {
            ai = -(double(i)-double(wr))*sint + (double(j)-double(wr))*cost;
            bi =  (double(i)-double(wr))*cost + (double(j)-double(wr))*sint;
            if((ai*ai*ira2 + bi*bi*irb2) > 1) continue;
            for(size_t n=0; n < 5; n++)
                x[n] += pow(ai, double(n));
        }
    for(size_t i=0; i < 3; i++)
        for(size_t j = i; j < i+3; j++) {
            A.at<float>(i, j-i) = x[j];
        }
    A = A.inv(DECOMP_SVD);
    for(size_t i = 0; i <= 2*wr; i++)
        for(size_t j = 0; j <= 2*wr; j++) {
            ai = -(double(i)-double(wr))*sint + (double(j)-double(wr))*cost;
            bi =  (double(i)-double(wr))*cost + (double(j)-double(wr))*sint;
            if((ai*ai*ira2 + bi*bi*irb2) > 1) continue;
            for(size_t n=0; n < 3; n++)
                y.at<float>(n,0) = pow(ai, double(n));
            y = A*y;
            kernel.at<float>(j,i) = y.at<float>(0,0);
        }
}

void
multiscalePb(const cv::Mat & image,
             cv::Mat & mPb_max,
             vector<vector<cv::Mat> > & gradients)
{
    cv::Mat kernel, angles, temp;
    vector<cv::Mat> layers, mPb_all;
    int n_ori = 8;
    int radii[4] = {3, 5, 10, 20};
    double* weights, *ori;

    weights = _mPb_Weights(image.channels());
    layers.resize(3);
    if(image.channels() == 3)
        cv::split(image, layers);
    else
        for(size_t i=0; i<3; i++)
            image.copyTo(layers[i]);

    cout<<"mPb computation commencing ..."<<endl;
    pb_parts_final_selected(layers, gradients);

    mPb_all.resize(n_ori);
    ori = cv::standard_filter_orientations(n_ori, RAD);
    for(size_t idx=0; idx<n_ori; idx++) {
        mPb_all[idx] = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
        for(size_t ch = 0; ch<gradients.size(); ch++) {
            MakeFilter(radii[ch-(ch/3)*3+int(ch>2)], ori[idx], kernel);
            cv::filter2D(gradients[ch][idx], gradients[ch][idx], CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
            cv::addWeighted(mPb_all[idx], 1.0, gradients[ch][idx], weights[ch], 0.0, mPb_all[idx]);
        }

        if(idx == 0) {
            angles = cv::Mat::ones(image.rows, image.cols, CV_32FC1);
            cv::multiply(angles, angles, angles, ori[idx]);
            mPb_all[idx].copyTo(mPb_max);
        }
        else
            for(size_t i=0; i<image.rows; i++)
                for(size_t j=0; j<image.cols; j++)
                    if(mPb_max.at<float>(i,j) < mPb_all[idx].at<float>(i,j)) {
                        angles.at<float>(i,j) = ori[idx];
                        mPb_max.at<float>(i,j) = mPb_all[idx].at<float>(i,j);
                    }
    }
    nonmax_oriented_2D(mPb_max, angles, temp, M_PI/8.0);
    temp.copyTo(mPb_max);

    //clean up
    delete[] weights;
    delete[] ori;
    layers.clear();
    mPb_all.clear();
}

void gPb_gen(const cv::Mat & mPb_max,
             const double* weights,
             const vector<cv::Mat> & sPb,
             vector<vector<cv::Mat> > & gradients,
             vector<cv::Mat> & gPb_ori,
             cv::Mat & gPb_thin,
             cv::Mat & gPb)
{
    cout<<"gPb computation commencing ... "<<endl;
    cv::Mat img_tmp, eroded, temp, bwskel;
    int n_ori = 8, nnz = 0;
    gradients.push_back(sPb);

    gPb_ori.resize(n_ori);
    for(size_t idx=0; idx<n_ori; idx++) {
        gPb_ori[idx] = cv::Mat::zeros(mPb_max.rows, mPb_max.cols, CV_32FC1);
        for(size_t ch=0; ch<gradients.size(); ch++)
            cv::addWeighted(gPb_ori[idx], 1.0, gradients[ch][idx], weights[ch], 0.0, gPb_ori[idx]);

        if(idx == 0)
            gPb_ori[idx].copyTo(gPb);
        else
            for(size_t i=0; i<mPb_max.rows; i++)
                for(size_t j=0; j<mPb_max.cols; j++)
                    if(gPb.at<float>(i,j) < gPb_ori[idx].at<float>(i,j))
                        gPb.at<float>(i,j) = gPb_ori[idx].at<float>(i,j);
    }

    gPb.copyTo(gPb_thin);
    for(size_t i=0; i<mPb_max.rows; i++)
        for(size_t j=0; j<mPb_max.cols; j++)
            if(mPb_max.at<float>(i,j)<0.05)
                gPb_thin.at<float>(i,j) = 0.0;

    bwskel = cv::Mat::zeros(mPb_max.rows, mPb_max.cols, CV_32FC1);
    gPb_thin.copyTo(img_tmp);
    do {
        cv::erode(img_tmp, eroded, cv::Mat(), cv::Point(-1, -1));
        cv::dilate(eroded, temp, cv::Mat(), cv::Point(-1,-1));
        cv::subtract(img_tmp, temp, temp);
        nnz = 0;
        for(size_t i=0; i<mPb_max.rows; i++)
            for(size_t j=0; j<mPb_max.cols; j++) {
                if(bwskel.at<float>(i,j) > 0.0 || temp.at<float>(i,j) > 0.0)
                    bwskel.at<float>(i,j) = 1.0;
                else
                    bwskel.at<float>(i,j) = 0.0;
                if(eroded.at<float>(i,j) != 0.0) nnz++;
            }
        eroded.copyTo(img_tmp);
    } while(nnz);
    cv::multiply(gPb_thin, bwskel, gPb_thin, 1.0);
}

void sPb_gen(cv::Mat & mPb_max,
             vector<cv::Mat> & sPb)
{
    cout<<"sPb computation commencing ... "<<endl;
    double **W, *D;
    int n_ori = 8, nnz;
    sPb.resize(n_ori);

    vector<cv::Mat> sPb_raw;
    cv::buildW(mPb_max, W, nnz, D);
    cv::normalise_cut(W, nnz, mPb_max.rows, mPb_max.cols, D, 17, sPb_raw);

    vector<cv::Mat> oe_filters;
    cv::gaussianFilters(n_ori, 1.0, 1, HILBRT_OFF, 3.0, oe_filters);

    for(size_t i=0; i<n_ori; i++) {
        sPb[i] = cv::Mat::zeros(mPb_max.rows, mPb_max.cols, CV_32FC1);
        for(size_t j=0; j<sPb_raw.size(); j++) {
            cv::Mat temp_blur;
            cv::filter2D(sPb_raw[j], temp_blur, CV_32F, oe_filters[i],
                         cv::Point(-1,-1), 0.0, cv::BORDER_REFLECT);
            cv::addWeighted(sPb[i], 1.0, cv::abs(temp_blur), 1.0, 0.0, sPb[i]);
            temp_blur.release();
        }
    }
    //clean up
    oe_filters.clear();
    sPb_raw.clear();
    delete[] W;
    delete[] D;
}

void
globalPb(const cv::Mat & image,
         cv::Mat & gPb,
         cv::Mat & gPb_thin,
         vector<cv::Mat> & gPb_ori)
{
    gPb = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
    cv::Mat mPb_max;
    vector<cv::Mat> sPb;
    vector<vector<cv::Mat> > gradients;
    double *weights;
    weights = _gPb_Weights(image.channels());

    //multiscalePb - mPb
    multiscalePb(image, mPb_max, gradients);
    //mPb_max.copyTo(gPb);

    //spectralPb   - sPb
    sPb_gen(mPb_max, sPb);

    //globalPb - gPb
    gPb_gen(mPb_max, weights, sPb, gradients, gPb_ori, gPb_thin, gPb);
    //clean up
    sPb.clear();
    gradients.clear();
    delete[] weights;
}
}
