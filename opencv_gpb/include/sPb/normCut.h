#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

namespace cv {
void normalise_cut(double **T,
                   int tlen,
                   int rows,
                   int cols,
                   double *D,
                   int nev,
                   std::vector<cv::Mat> & sPb_raw);
}
