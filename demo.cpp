#include <nuke/atomic.hpp>
#include <iostream>

int main()
{
  nuke::atomic_int a(0);

  #pragma omp parallel for
  for(int i = 0; i < 1024; ++i)
  {
    a++;
  }

  std::cout << "result after increment: " << (int)a << std::endl;

  #pragma omp parallel for
  for(int i = 0; i < 1024; ++i)
  {
    --a;
  }

  std::cout << "result after decrement: " << (int)a << std::endl;

  int racy(0);

  #pragma omp parallel for
  for(int i = 0; i < 1024; ++i)
  {
    racy++;
  }

  std::cout << "racy result after increment: " << racy << std::endl;

  return 0;
}

