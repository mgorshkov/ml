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

#include <np/Array.hpp>
#include <ml/sklearn/neighbors/WeightsType.hpp>
#include <ml/sklearn/neighbors/AlgorithmType.hpp>
#include <ml/sklearn/metrics/DistanceMetricType.hpp>

namespace ml {
    namespace sklearn {
        namespace neighbors {
            using Size = np::Size;

            template<typename DType, Size SizeT, Size... SizeTs>
            using Array = np::Array<DType, SizeT, SizeTs...>;

            /// Classifier implementing the k-nearest neighbors vote.
            template <typename DataType, typename TargetType, np::Size FeatureCount>
            class KNeighborsClassifier {
            public:
                KNeighborsClassifier(int n_neighbors = 5,
                                     WeightsType weights = WeightsType::kUniform,
                                     AlgorithmType algorithm = AlgorithmType::kAuto,
                                     int leaf_size = 30,
                                     int p = 2,
                                     metrics::DistanceMetricType metric = metrics::DistanceMetricType::kMinkowski) {
                    // generate n_neighbors random points
                }

                // Fit the k-nearest neighbors classifier from the training dataset.
                // X - training data
                // y - target values
                template <Size Count>
                void fit(const Array<DataType, Count, FeatureCount>& X, const Array<TargetType, Count>& y) {

                }

                // Predict the class labels for the provided data.
                // X - Test samples.
                template <Size Count>
                Array<TargetType, Count> predict(const Array<DataType, Count, FeatureCount>& X) {

                }
            private:

            };
        }
    }
}
