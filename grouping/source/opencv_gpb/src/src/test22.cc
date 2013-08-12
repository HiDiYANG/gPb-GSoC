#include <iostream>
using namespace std;

void test(int *array)
{
  int *test = array;
  for(size_t i = 0; i<3; i++)
    cout<<"array["<<i<<"]: "<<test[i]<<endl;
}

int main(int argc, char** argv)
{
  int array[3] = {1, 2, 3};
  test(array);
}
