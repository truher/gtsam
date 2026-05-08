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

  const double initialRotation = M_PI / 4.0;
  Rot2 R1(initialRotation);
  double error = 0.1;
  Rot2 R2(initialRotation + trueOmega * deltaT - error);

  EXPECT(assert_equal(error, factor.evaluateError(R1, R2, B1)(0), 1e-6))

  Values values;
  values.insert(R(1), R1);
  values.insert(R(2), R2);
  values.insert(B(1), B1);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

TEST(PlanarGyroFactor, optimize) {
  using noiseModel::Diagonal;
  const double trueOmega = 0.1;
  const double bias = 1;  // large !
  const double measuredOmega = trueOmega + bias;

  Rot2 R0(0);

  NonlinearFactorGraph graph;
  Values values;

  // Rotation prior.
  graph.add(PriorFactor<Rot2>(R(0), Rot2(), Diagonal::Sigmas(Vector1(0.001))));

  // Bias prior.
  graph.add(PriorFactor<double>(B(0), 1.0, Diagonal::Sigmas(Vector1(0.05))));

  // Bias evolution.
  SharedDiagonal biasNoise = Diagonal::Sigmas(Vector1(3e-6));
  graph.add(BetweenFactor<double>(B(0), B(1), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(1), B(2), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(2), B(3), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(3), B(4), 0, biasNoise));

  // Bias initial values.
  values.insert(B(0), 0.0);
  values.insert(B(1), 0.0);
  values.insert(B(2), 0.0);
  values.insert(B(3), 0.0);
  values.insert(B(4), 0.0);

  // Gyro measurements.
  double arw = 1e-8;
  double dt = 1.0;
  graph.add(PlanarGyroFactor(R(0), R(1), B(0),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(R(1), R(2), B(1),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(R(2), R(3), B(2),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(R(3), R(4), B(3),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));

  // Rotation initial values
  values.insert(R(0), Rot2());
  values.insert(R(1), Rot2());
  values.insert(R(2), Rot2());
  values.insert(R(3), Rot2());
  values.insert(R(4), Rot2());

  LevenbergMarquardtParams params;
  // default is 1e-5, 1e-6 is required to pass
  params.setAbsoluteErrorTol(1e-6);
  LevenbergMarquardtOptimizer optimizer(graph, values, params);
  Values result = optimizer.optimize();

  // Rotation increments are correct (true omega * dt).
  EXPECT(assert_equal(Rot2(0.0), result.at<Rot2>(R(0)), 1e-3));
  EXPECT(assert_equal(Rot2(0.1), result.at<Rot2>(R(1)), 1e-3));
  EXPECT(assert_equal(Rot2(0.2), result.at<Rot2>(R(2)), 1e-3));
  EXPECT(assert_equal(Rot2(0.3), result.at<Rot2>(R(3)), 1e-3));
  EXPECT(assert_equal(Rot2(0.4), result.at<Rot2>(R(4)), 1e-3));

  // Bias is the correct constant.
  EXPECT(assert_equal(1.0, result.at<double>(B(0)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(1)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(2)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(3)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(4)), 1e-3));

  Marginals marginals(graph, result);

  // Rotation variance grows superlinearly due to bias random walk,
  // i.e. bias variance compounds.
  EXPECT(assert_equal(0.000001, marginals.marginalCovariance(R(0))(0, 0), 1e-6))
  EXPECT(assert_equal(0.002501, marginals.marginalCovariance(R(1))(0, 0), 1e-6))
  EXPECT(assert_equal(0.010001, marginals.marginalCovariance(R(2))(0, 0), 1e-6))
  EXPECT(assert_equal(0.022501, marginals.marginalCovariance(R(3))(0, 0), 1e-6))
  EXPECT(assert_equal(0.040001, marginals.marginalCovariance(R(4))(0, 0), 1e-6))

  // Bias variance is constant.
  EXPECT(assert_equal(0.002500, marginals.marginalCovariance(B(0))(0, 0), 1e-6))
  EXPECT(assert_equal(0.002500, marginals.marginalCovariance(B(1))(0, 0), 1e-6))
  EXPECT(assert_equal(0.002500, marginals.marginalCovariance(B(2))(0, 0), 1e-6))
  EXPECT(assert_equal(0.002500, marginals.marginalCovariance(B(3))(0, 0), 1e-6))
  EXPECT(assert_equal(0.002500, marginals.marginalCovariance(B(4))(0, 0), 1e-6))
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
