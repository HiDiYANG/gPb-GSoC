#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
#include <time.h>
#include <vector>

using namespace cv;
using namespace std;

namespace test
{
  class parallelTestBody : public ParallelLoopBody{
  public:
    parallelTestBody(vector<Mat> & _src){src = &_src;}
    void operator()(const Range & range) const
    {
      vector<Mat> & srcMat = *src;
      int rows = srcMat[0].rows;
      int cols = srcMat[0].cols;
 for (int colIdx = range.start; colIdx < range.end; colIdx++)
      for(size_t i=0; i<rows; i++ )
	for(size_t j=0; j<cols; j++)
	  for (int n = 0; n < 8; n++){

	    srcMat[n].at<float>(i,j) = pow(srcMat[n].at<float>(i,j)+1, 3);
	  }
    }
  private:
    vector<Mat>* src;
  };
}

void parallelTestWithFor(vector<Mat> & srcMat)
{
  int rows = srcMat[0].rows;
  int cols = srcMat[0].cols;
  for(size_t n=0; n<3; n++)
  for(size_t i=0; i<rows; i++ )
    for(size_t j=0; j<cols; j++)
      for (int colIdx = 0; colIdx < srcMat.size(); colIdx++){
	srcMat[colIdx].at<float>(i,j) = pow(srcMat[colIdx].at<float>(i,j)+1, 3);
      }	
}

void parallelTestWithParallel_for_(vector<Mat> & srcMat){
  int totalCols = 3;
  parallel_for_(Range(0, totalCols), test::parallelTestBody(srcMat));
}

int main(int argc, char* argv[])
{
  vector<Mat> testInput;
  testInput.resize(8);
  for(size_t i=0; i<8; i++)
    testInput[i] = Mat::ones(400,400, CV_32F);
  
  clock_t start, stop;

  start = clock();
  parallelTestWithFor(testInput);
  stop = clock();
  cout<<"Running time using \'for\':"<<(double)(stop - start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;

  start = clock();
  parallelTestWithParallel_for_(testInput);
  stop = clock();
  cout<<"Running time using \'parallel_for_\':"<<(double)(stop - start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
}
