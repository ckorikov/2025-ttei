#include <iostream>
#include <random>
#include <vector>

#include <ttie/ttie.h>

using namespace ttie;

int main()
{
//     Tensor a(std::vector<float>({4, 3, 2, 1}), true);
//     Tensor b(std::vector<float>({1, 2, 3, 4}), true);
//     a.reshape({1, 2, 2});
//     b.reshape({1, 2, 2});
//
//     Tensor c = Tensor::log(a);
//
//     std::cout << "c:" << std::endl;
//     c.backward();
//     std::cout << "backwarded" << std::endl;
//     std::cout << a[{0, 1, 0}] << std::endl;
//     std::cout << a << std::endl;
//     std::cout << c << std::endl;

    Tensor t;
    std::cout << t.empty();
    return 0;
}
