#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
using namespace cv;
namespace test
{
  Mat parallelTestInvoker(const BlockedRange & range, Mat& _src)
  {
    Mat src;
    _src.copyTo(src);
    for (int colIdx = range.begin(); colIdx < range.end(); colIdx++)
      for (int i = 0; i < src.rows; ++i)
	src.at<float>(i, colIdx) = std::pow(src.at<float>(i, colIdx),3);
    return src;
  }
}//namesapce test

void parallelTestWithFor(Mat & src)//'for' loop
{
  for (int x = 0; x < src.cols; ++x)
    for (int y = 0; y < src.rows; ++y)
      src.at<float>(y, x) = std::pow(src.at<float>(y, x),3);
}

void parallelTestWithParallel_for(Mat & src)//'parallel_for' loop
{
  int totalCols = src.cols;
  BlockedRange range(0, totalCols);
  parallel_for(range, test::parallelTestInvoker(range, src));
};

//namespace cv
