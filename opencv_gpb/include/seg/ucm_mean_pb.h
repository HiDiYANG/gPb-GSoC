#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <deque>
#include <queue>
#include <vector>
#include <list>
#include <map>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#define DOUBLE_SIZE 1
#define SINGLE_SIZE 0

namespace cv {
void ucm_mean_pb(const cv::Mat & input1,
                 const cv::Mat & input2,
                 cv::Mat & output,
                 bool label);
}
