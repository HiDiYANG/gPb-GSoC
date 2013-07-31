//#include "globalPb.hh"
//#include "cv_lib_filters.hh"
//#include "contour2ucm.hh"
#include <iostream>
#include <list>
#include <stack>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//#include <opencv2/core/core.hpp>



using namespace std;

bool my_fun(int* w)
{
  cout<<"w: "<<*w<<endl;
  (*w)++;
  cout<<"w: "<<*w<<endl;
  return true;
}


int main()
{
  int w = 10;
  bool flag = my_fun(&w);
  cout<<"flag: "<<flag<<" , w: "<<w<<endl;
}
