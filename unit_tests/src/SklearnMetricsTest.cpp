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

#include <gtest/gtest.h>
#include <iostream>

#include <ml/sklearn/metrics/DistanceMetric.hpp>
#include <ml/sklearn/metrics/EuclideanDistance.hpp>

using namespace ml::sklearn::metrics;

class SklearnMetricsTest : public ::testing::Test {
protected:

};

TEST_F(SklearnMetricsTest, euclideanDistanceTest) {
/*
>>> from sklearn.metrics import DistanceMetric
>>> dist = DistanceMetric.get_metric('euclidean')
>>> X = [[0, 1, 2],
        [3, 4, 5]]
>>> dist.pairwise(X)
array([[ 0.        ,  5.19615242],
      [ 5.19615242,  0.        ]])
*/
    auto dist = DistanceMetric<np::float_>::get_metric(DistanceMetricType::kEuclidean);
    double array_c[2][3] = {{0, 1, 2}, {3, 4, 5}};
    np::Array<np::float_> array{array_c};
    auto result = dist->pairwise(array);
    double result_array_c[2][2] = {{0., 5.19615242}, {5.19615242, 0.}};
    np::Array<np::float_> result_sample{result_array_c};
    auto equal = result == result_sample;
    EXPECT_TRUE(equal);
}

