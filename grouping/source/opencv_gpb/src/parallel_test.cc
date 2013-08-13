#include <time.h>
#include "parallel_test.hh"
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
  Mat testInput = Mat::ones(2, 400, CV_32F);
  clock_t start, stop;

  start = clock();
  parallelTestWithFor(testInput);
  stop = clock();
  cout<<"Running time using \'for\':"<<(double)(stop - start)/CLOCKS_PER_SEC<<"s"<<endl;

  start = clock();
  parallelTestWithParallel_for(testInput);
  stop = clock();
  cout<<"Running time using \'parallel_for\':"<<(double)(stop - start)/CLOCKS_PER_SEC<<"s"<<endl;
}
