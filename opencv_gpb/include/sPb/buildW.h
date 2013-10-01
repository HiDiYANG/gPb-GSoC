#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include "array.h"
#include "smatrix.h"
#include "affinity.h"
#include "ic.h"


namespace cv
{
void buildW(const cv::Mat & input, double** &T, int & wz, double* &D);
}
