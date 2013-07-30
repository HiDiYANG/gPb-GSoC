//#include "globalPb.hh"
//#include "cv_lib_filters.hh"
//#include "contour2ucm.hh"
#include <iostream>
#include <list>
#include <stack>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>



using namespace std;

cv::Point show(int x, int y)
{
  return cv::Point(x, y);
}

class aaa{
public:
  int x;
  int y;

  aaa(): x(0), y(0){};
};

int main()
{
  int* a;
 
  aaa* c = new aaa[3];
  c[0].x = 10; c[0].y = 10;
  c[1].x = 20; c[1].y = 20;
  cout<<c[1].x<<", "<<c[1].y<<endl;
  
  aaa bbb = c[0];
  cout<<bbb.x<<", "<<bbb.y<<endl;
  
  
 

  stack<int> q;
  stack<int> q2;
  for(size_t i=0; i<10; i++)
  {
    q.push(i);
    q2.push(i*2);
  }  

  cout<<"here"<<endl;

  stack<int>* q_st;
  q_st->push(*q);
  q_st->push(*q2);
  //stack<int>& q_o1 = *q_st.top();
  /*while(!q_o1.empty()){
    cout<<q_o1.top()<<endl;
    q_o1.pop();
    }*/




  //stack<aaa> q3;
  //q3.push(*c);
  //cout<<"q3.x: "<<q3.top().x<<", q3.y: "<<q3.top().y<<endl;
  

  cv::Point aa = show(10, 10);
  cout<<"aa.x: "<<aa.x<<", aa.y: "<<aa.y<<endl;

}
