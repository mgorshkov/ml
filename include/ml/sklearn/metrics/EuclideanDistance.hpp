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

#pragma once

#include <memory>

#include <np/Array.hpp>
#include <ml/sklearn/metrics/Distance.hpp>

namespace ml {
    namespace sklearn {
        namespace metrics {
            template <typename DType, np::Size... Sizes>
            class EuclideanDistance;

            //np.sqrt(np.sum(np.square(X - Y))
            template <typename DType, np::Size NSamples1, np::Size NSamples2, np::Size NFeatures>
            class EuclideanDistance<DType, NSamples1, NSamples2, NFeatures> : public Distance<DType, NSamples1, NSamples2, NFeatures> {
            public:
                virtual np::Array<DType, NSamples1, NSamples1> pairwise(const np::Array<DType, NSamples1, NFeatures>& X) {
                    //dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
                    return sqrt(dot(X, X) - 2 * dot(X, X) + dot(X, X));
                }

                virtual np::Array<DType, NSamples1, NSamples2> pairwise(const np::Array<DType, NSamples1, NFeatures>& X, const np::Array<DType, NSamples2, NFeatures>& Y) {
                    //dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
                    return sqrt(dot(X, X) - 2 * dot(X, Y) + dot(Y, Y));
                }
            };

            template <typename DType>
            class EuclideanDistance<DType> : public Distance<DType> {
            public:
                virtual np::Array<DType> pairwise(const np::Array<DType>& X) {
                    if (X.shape().size() != 2) {
                        throw std::runtime_error("2D array expected");
                    }
                }
            };

            template <typename DType, np::Size... SizeTs>
            using EuclideanDistancePtr = std::shared_ptr<EuclideanDistance<DType, SizeTs...>>;
        }
    }
}
