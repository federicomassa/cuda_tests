#include <iostream>
#include <sstream>
#include <cstdlib>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "ERROR - 1 argument needed" << std::endl;
    exit(1);
  }
  
  std::stringstream ss;
  ss << argv[1];

  int N;
  ss >> N;

  std::cout << __LINE__ << std::endl;
  int a[N], b[N], c[N];
  std::cout << __LINE__ << std::endl;
  
  return 0;
}
