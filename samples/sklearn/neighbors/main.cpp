/*
ML Methods on top of NP library
Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>

#include <ml/sklearn/datasets/datasets.hpp>
#include <ml/sklearn/neighbors/KNeighborsClassifier.hpp>

int main(int, char **) {
    using namespace ml::sklearn::datasets;
    using namespace ml::sklearn::neighbors;

    auto iris = load_iris();
    auto data = iris["data"];
    auto target = iris["target"];

    auto kn = KNeighborsClassifier<np::float_, np::short_, 4>{};
    kn.fit(data[2:], target[2:]);
    std::cout << kn.predict([data[0], data[1]]) << std::endl;
    // array([0, 0])
    return 0;
}
