//#include "globalPb.hh"
//#include "cv_lib_filters.hh"
//#include "contour2ucm.hh"
#include <iostream>
#include <list>
#include <stack>
#include <vector>

using namespace std;

template<class T>
class Node
{  
public:
  Node* next;
  T data;
  
  Node(T d);
  void appendToTail(T d);
};

template<class T>
Node<T>::Node(T d)
{
  data = d; 
}

template<class T>
void Node<T>::appendToTail(T d)
{
  Node<T> end(d);
  Node<T> * n = this;
  while(n->next)
    n = n->next;
  n->next = &end;
}

int main()
{
  Node<int> lst(10);
  cout<<"lst data: "<<lst.data<<endl;
  lst.appendToTail(20);
  cout<<"lst data next address: "<<lst.next->data<<endl;
  
}
