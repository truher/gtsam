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
using symbol_shorthand::P;

TEST(PlanarGyroFactor, evaluateError) {
  std::cout << "evaluateError" << std::endl;

  PlanarGyroMeasurement measurement(0.01);

  const double trueOmega = M_PI / 10.0;
  const double B1 = 0.3;

  // Measurement includes bias.
  double measuredOmega = trueOmega + B1;
  double deltaT = 1.0;
  measurement.integrate(measuredOmega, deltaT);

  PlanarGyroFactor factor(P(1), P(2), B(1), measurement);

  const double initialRotation = M_PI / 4.0;
  Pose2 P1(0.0, 0.0, initialRotation);
  double error = 0.1;
  Pose2 P2(0.0, 0.0, initialRotation + trueOmega * deltaT - error);

  std::cout << "evaluateError A" << std::endl;

  EXPECT(assert_equal(Vector3(0, 0, error), factor.evaluateError(P1, P2, B1),
                      1e-6))

  Values values;
  values.insert(P(1), P1);
  values.insert(P(2), P2);
  values.insert(B(1), B1);

  std::cout << "evaluateError B" << std::endl;

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

TEST(PlanarGyroFactor, optimize) {
  std::cout << "optimize" << std::endl;
  using noiseModel::Diagonal;
  const double trueOmega = 0.1;
  const double bias = 1;  // large !
  const double measuredOmega = trueOmega + bias;
  std::cout << "optimize Z" << std::endl;

  NonlinearFactorGraph graph;
  Values values;

  std::cout << "optimize X" << std::endl;

  // Pose prior.
  graph.add(PriorFactor<Pose2>(P(0), Pose2(),
                               Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));

  std::cout << "optimize Y" << std::endl;

  // Translation evolution: constrains x and y, not theta
  // note if the theta sigma is too low, the solver thinks the system is
  // indeterminate.
  SharedDiagonal translationNoise = Diagonal::Sigmas(Vector3(1e-3, 1e-3, 1e-1));
  graph.add(BetweenFactor<Pose2>(P(0), P(1), Pose2(), translationNoise));
  graph.add(BetweenFactor<Pose2>(P(1), P(2), Pose2(), translationNoise));
  graph.add(BetweenFactor<Pose2>(P(2), P(3), Pose2(), translationNoise));
  graph.add(BetweenFactor<Pose2>(P(3), P(4), Pose2(), translationNoise));

  // Bias prior.
  graph.add(PriorFactor<double>(B(0), 1.0, Diagonal::Sigmas(Vector1(0.5))));

  std::cout << "optimize W" << std::endl;

  // Bias evolution.
  SharedDiagonal biasNoise = Diagonal::Sigmas(Vector1(3e-4));
  graph.add(BetweenFactor<double>(B(0), B(1), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(1), B(2), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(2), B(3), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(3), B(4), 0, biasNoise));

  std::cout << "optimize T" << std::endl;

  // Bias initial values.
  values.insert(B(0), 0.0);
  values.insert(B(1), 0.0);
  values.insert(B(2), 0.0);
  values.insert(B(3), 0.0);
  values.insert(B(4), 0.0);

  std::cout << "optimize U" << std::endl;

  // Gyro measurements.
  double arw = 1e-8;
  double dt = 1.0;
  graph.add(PlanarGyroFactor(P(0), P(1), B(0),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(P(1), P(2), B(1),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(P(2), P(3), B(2),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(P(3), P(4), B(3),
                             PlanarGyroMeasurement(arw, measuredOmega, dt)));
  std::cout << "optimize V" << std::endl;

  // // Rotation initial values
  values.insert(P(0), Pose2());
  values.insert(P(1), Pose2());
  values.insert(P(2), Pose2());
  values.insert(P(3), Pose2());
  values.insert(P(4), Pose2());

  std::cout << "optimize A" << std::endl;

  // graph.print();

  LevenbergMarquardtParams params;
  // default is 1e-5, 1e-6 is required to pass
  params.setAbsoluteErrorTol(1e-6);
  params.setVerbosityLM("SUMMARY");
  LevenbergMarquardtOptimizer optimizer(graph, values, params);
  Values result = optimizer.optimize();

  std::cout << "optimize B" << std::endl;

  // Rotation increments are correct (true omega * dt).
  EXPECT(assert_equal(Pose2(0, 0, 0.0), result.at<Pose2>(P(0)), 1e-3));
  EXPECT(assert_equal(Pose2(0, 0, 0.1), result.at<Pose2>(P(1)), 1e-3));
  EXPECT(assert_equal(Pose2(0, 0, 0.2), result.at<Pose2>(P(2)), 1e-3));
  EXPECT(assert_equal(Pose2(0, 0, 0.3), result.at<Pose2>(P(3)), 1e-3));
  EXPECT(assert_equal(Pose2(0, 0, 0.4), result.at<Pose2>(P(4)), 1e-3));

  // Bias is the correct constant.
  EXPECT(assert_equal(1.0, result.at<double>(B(0)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(1)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(2)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(3)), 1e-3));
  EXPECT(assert_equal(1.0, result.at<double>(B(4)), 1e-3));

  std::cout << "optimize C" << std::endl;

  Marginals marginals(graph, result);

  std::cout << "optimize D" << std::endl;
  // marginals.print();

  // Look at std devs because it's not so tiny
  // Rotation variance grows superlinearly due to bias random walk,
  // i.e. bias variance compounds.
  EXPECT(assert_equal(
      Vector3(0.010, 0.010, 0.010),
      Vector3(marginals.marginalCovariance(P(0)).diagonal().cwiseSqrt()), 1e-3))
  EXPECT(assert_equal(
      Vector3(0.010, 0.010, 0.050),
      Vector3(marginals.marginalCovariance(P(1)).diagonal().cwiseSqrt()), 1e-3))
  EXPECT(assert_equal(
      Vector3(0.010, 0.010, 0.100),
      Vector3(marginals.marginalCovariance(P(2)).diagonal().cwiseSqrt()), 1e-3))
  EXPECT(assert_equal(
      Vector3(0.010, 0.010, 0.150),
      Vector3(marginals.marginalCovariance(P(3)).diagonal().cwiseSqrt()), 1e-3))
  EXPECT(assert_equal(
      Vector3(0.010, 0.010, 0.200),
      Vector3(marginals.marginalCovariance(P(4)).diagonal().cwiseSqrt()), 1e-3))

  std::cout << "optimize E" << std::endl;

  // Bias variance is constant.
  EXPECT(
      assert_equal(0.05, sqrt(marginals.marginalCovariance(B(0))(0, 0)), 1e-3))
  EXPECT(
      assert_equal(0.05, sqrt(marginals.marginalCovariance(B(1))(0, 0)), 1e-3))
  EXPECT(
      assert_equal(0.05, sqrt(marginals.marginalCovariance(B(2))(0, 0)), 1e-3))
  EXPECT(
      assert_equal(0.05, sqrt(marginals.marginalCovariance(B(3))(0, 0)), 1e-3))
  EXPECT(
      assert_equal(0.05, sqrt(marginals.marginalCovariance(B(4))(0, 0)), 1e-3))
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
