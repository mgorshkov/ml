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

#include <ml/sklearn/metrics/DistanceMetric.hpp>
#include <ml/sklearn/metrics/DistanceMetricType.hpp>
#include <ml/sklearn/neighbors/AlgorithmType.hpp>
#include <ml/sklearn/neighbors/WeightsType.hpp>
#include <np/Array.hpp>
#include <scipy/stats/mode.hpp>

namespace ml {
    namespace sklearn {
        namespace neighbors {
            using Size = np::Size;

            template<typename DType = np::DTypeDefault, Size SizeT = np::SIZE_DEFAULT, Size... Sizes>
            using Array = np::Array<DType, SizeT, Sizes...>;

            using namespace scipy::stats;

            /// Classifier implementing the k-nearest neighbors vote.
            template <typename DataType, typename TargetType, np::Size... Sizes>
            class KNeighborsClassifier;

            template <typename DataType, typename TargetType, np::Size Count, np::Size FeatureCount>
            class KNeighborsClassifier<DataType, TargetType, Count, FeatureCount> {
            public:
                KNeighborsClassifier(int n_neighbors = 5,
                                     WeightsType weights = WeightsType::kUniform,
                                     AlgorithmType algorithm = AlgorithmType::kAuto,
                                     int leaf_size = 30,
                                     int p = 2,
                                     metrics::DistanceMetricType metric = metrics::DistanceMetricType::kMinkowski)
                    : m_neighbors{n_neighbors}
                    , m_metric{metric}{
                    if (algorithm != AlgorithmType::kAuto && algorithm != AlgorithmType::kBruteForce) {
                        throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                    }
                    if (weights != WeightsType::kUniform) {
                        throw std::runtime_error("Only Uniform weights are currently implemented");
                    }
                    (void)leaf_size;
                    (void)p;
                }

                // Fit the k-nearest neighbors classifier from the training dataset.
                // X - training data
                // y - target values
                void fit(const Array<DataType, Count, FeatureCount>& X, const Array<TargetType, Count>& y) {
                    m_X = X.copy();
                    m_y = y.copy();
                }

                // Predict the class labels for the provided data.
                // X - Test samples.
                Array<TargetType, Count> predict(const Array<DataType, Count, FeatureCount>& X) {
                    Array<TargetType, Count> y_pred;
                    auto metric = metrics::DistanceMetric<DataType>::get_metric(m_metric);

                    // Calculating distances
                    for (std::size_t i = 0; i < X.shape[0]; ++i) {  //  for every sample in X_test
                        Array<Array<DataType, Count>> distances;  // distance
                        for (std::size_t j = 0; j < m_X.shape[0]; ++j) {
                            auto d = metric->pairwise(X["i, :"], m_X["j, :"]);
                            distances.append((d, m_y[j]));
                        }
                        distances.sort(); // sorting distances in ascending order

                        // Getting k nearest neighbors
                        Array<Array<DataType, Count>> neighbors;
                        for (auto item = 0; item < m_neighbors; ++item) {
                            neighbors.append(distances[item][1]);  // appending K nearest neighbors
                        }

                        // Making predictions
                        y_pred.append(neighbors[0][0]);
                    }
                    return y_pred;
                }

            private:
                int m_neighbors;
                metrics::DistanceMetricType m_metric;
                const Array<DataType, Count, FeatureCount> m_X;
                const Array<TargetType, Count> m_y;
            };

            template <typename DataType, typename TargetType>
            class KNeighborsClassifier<DataType, TargetType> {
            public:
                KNeighborsClassifier(int n_neighbors = 5,
                                     WeightsType weights = WeightsType::kUniform,
                                     AlgorithmType algorithm = AlgorithmType::kAuto,
                                     int leaf_size = 30,
                                     int p = 2,
                                     metrics::DistanceMetricType metric = metrics::DistanceMetricType::kMinkowski)
                        : m_neighbors{n_neighbors}
                        , m_metric{metric}{
                    if (algorithm != AlgorithmType::kAuto && algorithm != AlgorithmType::kBruteForce) {
                        throw std::runtime_error("Only BruteForce algorithm is currently implemented");
                    }
                    if (weights != WeightsType::kUniform) {
                        throw std::runtime_error("Only Uniform weights are currently implemented");
                    }
                    (void)leaf_size;
                    (void)p;
                }

                // Fit the k-nearest neighbors classifier from the training dataset.
                // X - training data
                // y - target values
                void fit(const Array<DataType>& X, const Array<TargetType>& y) {
                    m_X = X.copy();
                    m_y = y.copy();
                }

                // Predict the class labels for the provided data.
                // X - Test samples.
                Array<TargetType> predict(const Array<DataType>& X) {
                    Array<TargetType> y_pred;
                    auto metric = metrics::DistanceMetric<DataType>::get_metric(m_metric);

                    // Calculating distances
                    for (std::size_t i = 0; i < X.shape()[0]; ++i) {  //  for every sample in X_test
                        std::vector<Array<DataType>> distances;  // distance
                        for (std::size_t j = 0; j < m_X.shape()[0]; ++j) {
                            auto d = metric->pairwise(X["i, :"], m_X["j, :"]);
                            distances.push_back(Array<DataType>{d, m_y[j]});
                        }
                        distances.sort(); // sorting distances in ascending order

                        // Getting k nearest neighbors
                        Array<DataType> neighbors{{2, X.shape()}};
                        for (auto item = 0; item < m_neighbors; ++item) {
                            neighbors.distances[item][1];  // appending K nearest neighbors
                        }

                        // Making predictions
                        y_pred.append(mode(neighbors)[0][0]);
                    }
                    return y_pred;
                }

            private:
                int m_neighbors;
                metrics::DistanceMetricType m_metric;
                Array<DataType> m_X;
                Array<TargetType> m_y;
            };
        }
    }
}
