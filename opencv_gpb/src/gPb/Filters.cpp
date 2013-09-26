//
//    Filters:
//       An extended library of opencv gaussian-based filters.
//       contents:
//       1D multi-order gaussian filters (Option: Hilbert Transform)
//       2D multi-order anistropic gaussian filters (Option: Hilbert Transform)
//       2D central-surrouding gaussian filters
//       2D texton filters
//       texton executation
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 31/05/13.
//    Copyright (c) 2013 The Australian National University.
//    and Willow Garage inc.
//    All rights reserved.
//
//

#include "Filters.h"
using namespace std;

class DFTconvolver {
public:
    DFTconvolver(int num_bins, const cv::Mat &gaussian_filter) : num_bins_(num_bins) {
        width_ = cv::getOptimalDFTSize(num_bins + gaussian_filter.cols - 1);
        cv::Mat TempB;
        cv::copyMakeBorder(gaussian_filter, TempB, 0, 0, 0, width_ - gaussian_filter.cols - 1, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::dft(TempB, filter_dft_, cv::DFT_ROWS, TempB.rows);
    }
    
    void conv(cv::Mat & hist)
    {
        cv::Mat TempA;
        int r=hist.rows, c=hist.cols;
        
        cv::copyMakeBorder(hist, TempA, 0, 0, 0, width_ - hist.cols - 1, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::dft(TempA, TempA, cv::DFT_ROWS, TempA.rows);
        cv::mulSpectrums(TempA, filter_dft_, TempA, cv::DFT_ROWS, false);
        cv::dft(TempA, TempA, cv::DFT_INVERSE+cv::DFT_SCALE, TempA.rows);
        
        int W_o = (TempA.cols-c)/2;
        TempA(cv::Rect(W_o, 0, c, r)).copyTo(hist);
    }
    
private:
    int num_bins_;
    int width_;
    cv::Mat filter_dft_;
};

namespace cv
{
    /************************************
     * Hilbert Transform
     ************************************/
    
    void
    convolveDFT(const cv::Mat & inputA,
                const cv::Mat & inputB,
                cv::Mat & output,
                bool label)
    {
        bool flag = label? SAME_SIZE : EXPAND_SIZE;
        cv::Mat TempA, TempB;
        int r=inputA.rows, c=inputA.cols;
        inputA.copyTo(TempA);
        inputB.copyTo(TempB);
        
        int width = cv::getOptimalDFTSize(inputA.cols+inputB.cols-1);
        cv::copyMakeBorder(TempA, TempA, 0, 0, 0, width-TempA.cols-1, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(TempB, TempB, 0, 0, 0, width-TempB.cols-1, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::dft(TempA, TempA, cv::DFT_ROWS, TempA.rows);
        cv::dft(TempB, TempB, cv::DFT_ROWS, TempB.rows);
        cv::mulSpectrums(TempA, TempB, TempA, cv::DFT_ROWS, false);
        cv::dft(TempA, TempA, cv::DFT_INVERSE+cv::DFT_SCALE, TempA.rows);
        
        if(flag){
	  int W_o = (TempA.cols-c)/2;
	  TempA(cv::Rect(W_o, 0, c, r)).copyTo(output);
        }
        else
	  TempA.copyTo(output);
    }
    
    void
    hilbertTransform1D(const cv::Mat & input,
                       cv::Mat & output,
                       bool label)
    {
        cv::Mat temp;
        input.copyTo(temp);
        if(temp.cols != 1 && temp.rows != 1){
            cout<<"Input must be a 1D matrix"<<endl;
        }
        int length = (temp.rows > temp.cols)? temp.rows : temp.cols;
        if(input.cols == 1)
            cv::transpose(temp, temp);
        cv::Mat hilbert(1, length, CV_32FC1);
        int half_len = (length-1)/2;
        for(int i = 0; i < hilbert.cols; i++){
            int m = i-half_len;
            if( m % 2 == 0)
                hilbert.at<float>(0, i) = 0.0;
            else
                hilbert.at<float>(0, i) = 1.0/(M_PI*double(m));
        }
        convolveDFT(temp, hilbert, temp, label);
        if(input.cols == 1)
            cv::transpose(temp, temp);
        temp.copyTo(output);
        
        //clean up;
        temp.release();
        hilbert.release();
    }
    
    /***************************************
     * Standard orientation generation
     ***************************************/
    double*
    standard_filter_orientations(int n_ori,
                                 bool label)
    {
        bool flag = label? RAD : DEG;
        double* oris = new double[n_ori];
        double ori = 0.0;
        if(flag){
            double ori_step = (n_ori>0) ? (M_PI/double(n_ori)) : 0;
            for(size_t i=0; i<n_ori; i++, ori += ori_step)
                oris[i] = ori;
        }
        else{
            double ori_step = (n_ori>0) ? (180.0/double(n_ori)) : 0;
            for(size_t i=0; i<n_ori; i++, ori += ori_step)
                oris[i] = ori;
        }
        return oris;
    }
    
    /****************************************************
     * Distribution Normalize and Mean value shifting
     ****************************************************/
    void
    normalizeDistr(const cv::Mat & input,
                   cv::Mat & output,
                   bool label)
    {
        bool flag = label ? ZERO : NON_ZERO;
        input.convertTo(output, CV_32FC1);
        /* If required, zero-mean shift*/
        if(flag)
            output = output - float(mean(output)[0]);
        /* Distribution Normalized */
        output = output / cv::norm(output, NORM_L1);
    }
    
    /*******************************
     * Matrix Rotation
     *******************************/
    
    int
    supportRotated(int x,
                   int y,
                   double ori,
                   bool label)
    {
        double sin_ori, cos_ori, mag0, mag1;
        bool flag = label ? X_ORI : Y_ORI;
        if(flag){
            cos_ori = double(x)*cos(ori);
            sin_ori = double(y)*sin(ori);
        }
        else{
            cos_ori = double(y)*cos(ori);
            sin_ori = double(x)*sin(ori);
        }
        mag0 = fabs(cos_ori - sin_ori);
        mag1 = fabs(cos_ori + sin_ori);
        return int(((mag0 > mag1)? mag0 : mag1)+1.0);
    }
    
    void
    rotate_2D_crop(const cv::Mat & input,
                   cv::Mat & output,
                   double ori,
                   int len_cols,
                   int len_rows,
                   bool label)
    {
        bool flag = label? RAD:DEG;
        cv::Mat tmp;
        cv::Mat rotate_M = cv::Mat::zeros(2, 3, CV_32FC1);
        cv::Point center = cv::Point((input.cols-1)/2, (input.rows-1)/2);
        double angle;
        
        if(flag)
            angle = ori/M_PI*180.0;
        else
            angle = ori;
        
        rotate_M = cv::getRotationMatrix2D(center, angle, 1.0);
        
        /* Apply rotation transformation to a matrix */
        cv::warpAffine(input, tmp, rotate_M, input.size(), cv::INTER_LINEAR);
        
        /* Cropping */
        int border_rows = (input.rows - len_rows)/2;
        int border_cols = (input.cols - len_cols)/2;
        cv::Rect cROI(border_cols, border_rows, len_cols, len_rows);
        output = tmp(cROI);
    }
    
    void rotate_2D(const cv::Mat & input,
                   cv::Mat & output,
                   double ori,
                   bool label)
    {
        rotate_2D_crop(input, output, ori, input.cols, input.rows, label);
    }
    
    /***********************
     * Filters Generation
     ***********************/
    
    /* 1D multi-order gaussian filter generation */
    void
    gaussianFilter1D(int half_len,
                     double sigma,
                     int deriv,
                     bool label,
                     cv::Mat & output)
    {
        bool hlbrt = label? HILBRT_ON : HILBRT_OFF;
        int len = 2*half_len+1;
        cv::Mat ones = cv::Mat::ones(len, 1, CV_32F);
        output  = cv::getGaussianKernel(len, sigma, CV_32F);
        if(deriv == 1){
            for(int i=0; i<len; i++){
                output.at<float>(i) = output.at<float>(i)*double(half_len-i);
            }
        }
        else if(deriv == 2){
            for(int i=0; i<len; i++){
                double x = double(i-half_len);
                output.at<float>(i) = output.at<float>(i)*(x*x/sigma-1.0);
            }
        }
        if(hlbrt)
            hilbertTransform1D(output, output, SAME_SIZE);
        
        if(deriv > 0)
            normalizeDistr(output, output, ZERO);
        else
            normalizeDistr(output, output, NON_ZERO);
    }
    
    void
    gaussianFilter1D(double sigma,
                     int deriv,
                     bool hlbrt,
                     cv::Mat & output)
    {
        int half_len = int(sigma*3.0);
        gaussianFilter1D(half_len, sigma, deriv, hlbrt, output);
    }
    
    /* multi-order anistropic gaussian filter generation */
    void
    gaussianFilter2D(int half_len,
                     double ori,
                     double sigma_x,
                     double sigma_y,
                     int deriv,
                     bool hlbrt,
                     cv::Mat & output)
    {
        /* rotate support ROI */
        int len = 2*half_len+1;
        int half_len_rotate_x = supportRotated(half_len, half_len, ori, X_ORI);
        int half_len_rotate_y = supportRotated(half_len, half_len, ori, Y_ORI);
        int half_rotate_len = (half_len_rotate_x > half_len_rotate_y)? half_len_rotate_x : half_len_rotate_y;
        cv::Mat output_x, output_y;
        
        /*   Conduct Compution */
        gaussianFilter1D(half_rotate_len, sigma_x, 0,     HILBRT_OFF, output_x);
        gaussianFilter1D(half_rotate_len, sigma_y, deriv, hlbrt, output_y);
        output = output_x*output_y.t();
        rotate_2D_crop(output, output, ori, len, len, DEG);
        
        /*  Normalize  */
        if(deriv > 0)
            normalizeDistr(output, output, ZERO);
        else
            normalizeDistr(output, output, NON_ZERO);
    }
    
    
    void
    gaussianFilter2D(double ori,
                     double sigma_x,
                     double sigma_y,
                     int deriv,
                     bool hlbrt,
                     cv::Mat & output)
    {
        /* actual size of kernel */
        int half_len_x = int(sigma_x*3.0);
        int half_len_y = int(sigma_y*3.0);
        int half_len = (half_len_x>half_len_y)? half_len_x : half_len_y;
        gaussianFilter2D(half_len, ori, sigma_x, sigma_y, deriv, hlbrt, output);
    }
    
    /* Central-surrounding gaussian filter */
    void
    gaussianFilter2D_cs(int half_len,
                        double sigma_x,
                        double sigma_y,
                        double scale_factor,
                        cv::Mat & output)
    {
        double sigma_x_c = sigma_x/scale_factor;
        double sigma_y_c = sigma_y/scale_factor;
        cv::Mat output_cen, output_sur;
        gaussianFilter2D(half_len, 0.0, sigma_x_c, sigma_y_c, 0, HILBRT_OFF, output_cen);
        gaussianFilter2D(half_len, 0.0, sigma_x,   sigma_y,   0, HILBRT_OFF, output_sur);
        cv::addWeighted(output_sur, 1.0, output_cen, -1.0, 0.0, output);
        normalizeDistr(output, output, ZERO);
    }
    
    void
    gaussianFilter2D_cs(double sigma_x,
                        double sigma_y,
                        double scale_factor,
                        cv::Mat & output)
    {
        int half_len_x = int(sigma_x*3.0);
        int half_len_y = int(sigma_y*3.0);
        int half_len = (half_len_x>half_len_y)? half_len_x : half_len_y;
        gaussianFilter2D_cs(half_len, sigma_x, sigma_y, scale_factor, output);
    }
    
    /* A set of multi-order anistropic gaussian filters generation */
    void
    gaussianFilters(int n_ori,
                    double sigma,
                    int deriv,
                    bool hlbrt,
                    double enlongation,
                    vector<cv::Mat> & filters)
    {
        double sigma_x = sigma;
        double sigma_y = sigma/enlongation;
        double* oris;
        filters.resize(n_ori);
        oris = standard_filter_orientations(n_ori, DEG);
        for(size_t i=0; i<n_ori; i++)
            gaussianFilter2D(oris[i], sigma_x, sigma_y, deriv, hlbrt, filters[i]);
    }
    
    /* Even or odd gaussian multi-order gaussian filters generation */
    void
    oeFilters(int n_ori,
              double sigma,
              vector<cv::Mat> & filters,
              bool label)
    {
        bool flag = label ? OE_EVEN : OE_ODD;
        if(flag)
            gaussianFilters(n_ori, sigma, 2, HILBRT_OFF, 3.0, filters);
        else
            gaussianFilters(n_ori, sigma, 2, HILBRT_ON, 3.0, filters);
    }
    
    /* Texton Filters Generation */
    void
    textonFilters(int n_ori,
                  double sigma,
                  vector<cv::Mat> & filters)
    {
        vector<cv::Mat> even_filters;
        vector<cv::Mat> odd_filters;
        cv::Mat f_cs;
        filters.resize(2*n_ori+1);
        oeFilters(n_ori, sigma, even_filters, OE_EVEN);
        oeFilters(n_ori, sigma, odd_filters,  OE_ODD );
        gaussianFilter2D_cs(sigma, sigma, M_SQRT2, f_cs);
        
        for(size_t i=0; i<n_ori; i++){
            even_filters[i].copyTo(filters[i]);
            odd_filters[i].copyTo(filters[n_ori+i]);
        }
        f_cs.copyTo(filters[2*n_ori]);
        //clean up
        even_filters.clear();
        odd_filters.clear();
    }
    
    /*******************************
     * Texton Filters Executation
     *******************************/
    
    void
    textonRun(const cv::Mat & input,
              cv::Mat & output,
              int n_ori,
              int Kmean_num,
              double sigma_sm,
              double sigma_lg)
    {
        vector<cv::Mat> filters_small, filters_large, filters;
        cv::Mat blur, labels, k_samples;
        
        filters.resize(4*n_ori+2);
        textonFilters(n_ori, sigma_sm, filters_small);
        textonFilters(n_ori, sigma_lg, filters_large);
        
        for(size_t i=0; i<2*n_ori+1; i++){
            filters_small[i].copyTo(filters[i]);
            filters_large[i].copyTo(filters[2*n_ori+1+i]);
        }
        
        k_samples = cv::Mat::zeros(input.rows*input.cols, 4*n_ori+2, CV_32FC1);
        
        for(size_t idx=0; idx< 4*n_ori+2; idx++){
            cv::filter2D(input, blur, CV_32F, filters[idx], cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
            for(size_t i = 0; i<k_samples.rows; i++)
                k_samples.at<float>(i, idx) = blur.at<float>(i%blur.rows, i/blur.rows);
        }
        
        cv::kmeans(k_samples, Kmean_num, labels,
                   cv::TermCriteria(cv::TermCriteria::EPS, 10, 0.0001),
                   3, cv::KMEANS_PP_CENTERS);
        
        output = cv::Mat::zeros(blur.rows, blur.cols, CV_32SC1);
        for(size_t i=0; i<labels.rows; i++)
            output.at<int>(i%output.rows, i/output.rows)=labels.at<int>(i, 0);
        output.convertTo(output, CV_32FC1);
    }
    
    cv::Mat_<int>
    weight_matrix_disc(int r)
    {
        int size = 2*r + 1;
        int r_sq = r*r;
        cv::Mat_<int> weights = cv::Mat_<int>::zeros(size, size);
        for (int i = 0; i< weights.rows; i++)
            for (int j = 0; j< weights.cols; j++) {
                int x_sq = (i-r)*(i-r);
                int y_sq = (j-r)*(j-r);
                if ((x_sq + y_sq) <= r_sq)
                    weights(i, j) = 1;
            }
        weights(r, r) = 0;
        return weights;
    }
    
    /*
     * Construct orientation slice lookup map.
     */
    cv::Mat orientation_slice_map(int r,
				  int n_ori)
    {
        /* initialize map */
        int size = 2*r+1;
        cv::Mat slice_map = cv::Mat::zeros(size, size, CV_32FC1);
        for (int i = 0, y = size/2; i < size; i++, y--)
            for (int j = 0, x = -size/2; j < size; j++, x++) {
                double ori = atan2(double(y), double(x));
                slice_map.at<float>(i, j) = ori/M_PI*180.0;
            }
        return slice_map;
    }

    /** Unit of computation used
     */
    class ParallelInvokerUnit {
    private:
        double *oris_;
        DFTconvolver *convolver_left_;
        DFTconvolver *convolver_right_;
        int num_bins_;
        cv::Mat_<int> weights_;
        cv::Mat label_exp_;
        cv::Mat slice_map_;
        cv::Mat gaussian_kernel_;
        cv::Size label_size_;
        int r_;
    public:
        ParallelInvokerUnit(int num_bins, size_t n_ori, int r, const cv::Mat & label, const cv::Mat &gaussian_kernel) : 
num_bins_(num_bins), r_(r) {
            label_size_ = label.size();
            convolver_left_ = new DFTconvolver(num_bins, gaussian_kernel);
            convolver_right_ = new DFTconvolver(num_bins, gaussian_kernel);

            oris_ = standard_filter_orientations(n_ori, DEG);

            slice_map_ = orientation_slice_map(r, n_ori);

            weights_ = weight_matrix_disc(r);
            gaussian_kernel.copyTo(gaussian_kernel_);
            cv::copyMakeBorder(label, label_exp_, r, r, r, r, cv::BORDER_REFLECT);
        }

        cv::Mat_<float>
        operator() (const size_t &idx) {
            cv::Mat_<float> gradients = cv::Mat_<float>::zeros(label_size_);

            cv::Mat_<float> hist_left  = cv::Mat_<float>::zeros(1, num_bins_);
            cv::Mat_<float> hist_right = cv::Mat_<float>::zeros(1, num_bins_);

            for(int i=r_; i<label_exp_.rows-r_; i++)
                for(int j=r_; j<label_exp_.cols-r_; j++){
                    hist_left.setTo(0.0);
                    hist_right.setTo(0.0);
                    for(int x= -r_; x <= r_; x++)
                        for(int y= -r_; y <= r_; y++){
                            int bin = int(label_exp_.at<float>(i+x, j+y));
                            if(slice_map_.at<float>(x+r_, y+r_) > oris_[idx]-180.0 && 
                                slice_map_.at<float>(x+r_, y+r_) <= oris_[idx])
                                hist_right(0, bin) += double(weights_(x+r_, y+r_));
                            else
                                hist_left(0, bin) += double(weights_(x+r_, y+r_));
                        }
                    
                    convolver_right_->conv(hist_right);
                    convolver_left_->conv(hist_left);
                    
                    double sum_l = sum(hist_left)[0], sum_r = sum(hist_right)[0];
                    
                    double tmp = 0.0, tmp1 = 0.0, tmp2 = 0.0, hist_r, hist_l;
                    for(size_t nn = 0; nn<num_bins_; nn++){
                        if(sum_r == 0)
                            hist_r = hist_right(0,nn);
                        else
                            hist_r = hist_right(0,nn)/sum_r;
                        
                        if(sum_l == 0)
                            hist_l = hist_left(0,nn);
                        else
                            hist_l = hist_left(0,nn)/sum_l;
                        
                        tmp1 = hist_r-hist_l;
                        tmp2 = hist_r+hist_l;
                        if(tmp2 < 0.00001)
                            tmp2 = 1.0;
                        
                        tmp += 4.0*(tmp1*tmp1)/tmp2;
                    }
                    gradients.at<float>(i-r_,j-r_) = tmp;
                }
            return gradients;
        }
    };

    void
    gradient_hist_2D(const cv::Mat & label,
                     int r,
                     int n_ori,
                     int num_bins,
                     cv::Mat & gaussian_kernel,
                     vector<cv::Mat> & gradients)
    {
        ParallelInvokerUnit parallel_invoker_unit(num_bins, n_ori, r, label, gaussian_kernel);

        gradients.resize(n_ori);
        for(size_t idx = 0; idx < n_ori; idx++)
            gradients[idx] = parallel_invoker_unit(idx);
    }
    
    void
    gradient_hist_2D(const cv::Mat & label,
                     int r,
                     int n_ori,
                     int num_bins,
                     vector<cv::Mat> & gradients)
    {
      int length = 7;
      cv::Mat impulse_resp = cv::Mat::zeros(1, length, CV_32FC1);
      impulse_resp.at<float>(0, (length-1)/2) = 1.0;
      gradient_hist_2D(label, r, n_ori, num_bins, impulse_resp, gradients);
    }
    
    
    //-------------------- Parallel Computation attempt ------------------------
    
    struct parallelInvoker{
        const cv::Mat * label_ptr;
        int r;
        int num_bins;
        cv::Mat * gaussian_kernel_ptr;
        vector<cv::Mat> * gradients_ptr;
        
        void operator()(const cv::BlockedRange & range) const
        {  
            const cv::Mat & label = * label_ptr;
            cv::Mat & gaussian_kernel = * gaussian_kernel_ptr;
            vector<cv::Mat> & gradients = * gradients_ptr;
            
            gradients.resize(range.end());
            
            ParallelInvokerUnit parallel_invoker_unit(num_bins, range.end(), r, label, gaussian_kernel);
            for(size_t idx = range.begin(); idx < range.end(); idx++)
                gradients[idx] = parallel_invoker_unit(idx);
        }
    };
    
    void 
    parallel_for_gradient_hist_2D(const cv::Mat & label,
                                  int r,
                                  int n_ori,
                                  int num_bins,
                                  cv::Mat & gaussian_kernel,
                                  std::vector<cv::Mat> & gradients)
    {
        parallelInvoker parallel;
        parallel.label_ptr = & label;
        parallel.r = r;
        parallel.num_bins = num_bins;
        parallel.gaussian_kernel_ptr = & gaussian_kernel; 
        parallel.gradients_ptr = & gradients;
        int totalCols = n_ori;
        cv::BlockedRange range(0, totalCols);
        cv::parallel_for(range, parallel);
    }
}
