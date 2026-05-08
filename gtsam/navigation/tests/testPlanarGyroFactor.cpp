/**
 * @file testPlanarGyroFactor.cpp
 * @date May 1, 2026
 * @author joel@truher.org
 * @brief tests for PlanarGyroFactor
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/debug.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/navigation/PlanarGyroFactor.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/slam/BetweenFactor.h>

#include <cmath>
#include <list>
#include <memory>

#include "imuFactorTesting.h"

using namespace std::placeholders;
using namespace std;

namespace gtsam {

using symbol_shorthand::B;
using symbol_shorthand::R;

TEST(PlanarGyroFactor, evaluateError) {
  PlanarGyroMeasurement measurement(0.01);

  const double trueOmega = M_PI / 10.0;
  const double B1 = 0.3;

  // Measurement includes bias.
  double measuredOmega = trueOmega + B1;
  double deltaT = 1.0;
  measurement.integrate(measuredOmega, deltaT);

  PlanarGyroFactor factor(R(1), R(2), B(1), measurement);

  Rot2 R1(M_PI / 4.0);
  Rot2 R2(M_PI / 4.0 + M_PI / 10.0);
  // Estimate is correct.
  EXPECT(assert_equal(0.0, factor.evaluateError(R1, R2, B1)(0), 1e-6))

  Values values;
  values.insert(R(1), R1);
  values.insert(R(2), R2);
  values.insert(B(1), B1);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

TEST(PlanarGyroFactor, optimize) {
  using noiseModel::Diagonal;

  NonlinearFactorGraph graph;
  // Priors are zero
  graph.add(PriorFactor<Rot2>(R(1), Rot2(), Diagonal::Sigmas(Vector1(0.001))));
  graph.add(PriorFactor<double>(B(1), 0.0, Diagonal::Sigmas(Vector1(0.05))));

  PlanarGyroMeasurement measurement(0.01);
  measurement.integrate(M_PI / 20, 5);
  graph.add(PlanarGyroFactor(R(1), R(2), B(1), measurement));

  Values values;
  // Initial values are always zero
  values.insert(R(1), Rot2());
  values.insert(R(2), Rot2());
  values.insert(B(1), 0.0);

  LevenbergMarquardtOptimizer optimizer(graph, values);
  Values result = optimizer.optimize();
  EXPECT(assert_equal(M_PI / 4, result.at<Rot2>(R(2)).theta(), 1e-6))

  Marginals marginals(graph, result);
  Matrix1 varR1 = marginals.marginalCovariance(R(1));
  Matrix1 varR2 = marginals.marginalCovariance(R(2));
  Matrix1 varB1 = marginals.marginalCovariance(B(1));
  // prior
  EXPECT(assert_equal(0.0000010, varR1(0, 0), 1e-7))
  // ???
  EXPECT(assert_equal(0.1125010, varR2(0, 0), 1e-7))
  // prior
  EXPECT(assert_equal(0.0025000, varB1(0, 0), 1e-7))
}

TEST(PlanarGyroFactor, bodyPSensorWithBias) {
  using noiseModel::Diagonal;
  const double trueOmega = 0.1;
  const double bias = 1;  // large !
  const double measuredOmega = trueOmega + bias;

  Rot2 R0(0);

  NonlinearFactorGraph graph;
  graph.add(PriorFactor<Rot2>(R(0), Rot2(), Diagonal::Sigmas(Vector1(0.001))));
  graph.add(PriorFactor<double>(B(0), bias, Diagonal::Sigmas(Vector1(0.05))));

  Values values;
  values.insert(R(0), Rot2());
  values.insert(B(0), bias);

  SharedDiagonal biasNoise = Diagonal::Sigmas(Vector1(0.000003));

  int numRotations = 10;
  for (int i = 1; i < numRotations; i++) {

    PlanarGyroMeasurement measurement(1e-8);
    measurement.integrate(measuredOmega, 1.0);

    graph.add(PlanarGyroFactor(R(i - 1), R(i), B(i - 1), measurement));
    graph.add(BetweenFactor<double>(B(i - 1), B(i), 0, biasNoise));

    // Initial values are always zero
    values.insert(R(i), Rot2());
    values.insert(B(i), 0.0);
  }

  LevenbergMarquardtParams params;
  // default is 1e-5, 1e-6 is required to pass
  params.setAbsoluteErrorTol(1e-6);
  Values result = LevenbergMarquardtOptimizer(graph, values, params).optimize();

  for (int i = 0; i < numRotations; i++) {
    EXPECT(assert_equal(bias, result.at<double>(B(i)), 1e-3));
    EXPECT(assert_equal(Rot2(trueOmega * i), result.at<Rot2>(R(i)), 1e-3));
  }
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
