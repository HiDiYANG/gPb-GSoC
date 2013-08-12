#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
using namespace cv;

  struct parallelTestInvoker
  {
    Mat* src_ptr;

    void operator()(const cv::BlockedRange & range) const
    {
      Mat & src = *src_ptr;
      for (int colIdx = range.begin(); colIdx < range.end(); colIdx++)
	for (int i = 0; i < src.cols; i++)
	  for(int j =0; j < 50; j++)
	    src.at<float>(colIdx, i) = std::pow(src.at<float>(colIdx, i)+1,3);
    }
  };

void parallelTestWithFor(Mat & src)//'for' loop
{
  for (int x = 0; x < src.rows; x++)
    for (int y = 0; y < src.cols; y++)
      for(int cont = 0; cont < 50; cont++)
	src.at<float>(x, y) = std::pow(src.at<float>(x, y)+1,3);
};

void parallelTestWithParallel_for(Mat & src)//'parallel_for' loop
{
  parallelTestInvoker parallel;
  parallel.src_ptr = & src;
  int totalCols = src.rows;
  BlockedRange range(0, totalCols);
  parallel_for(range, parallel);
};

//namespace cv
