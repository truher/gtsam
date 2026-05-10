/**
 * @file testPlanarGyroFactor.cpp
 * @date May 1, 2026
 * @author joel@truher.org
 * @brief tests for PlanarGyroFactor
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/PlanarGyroFactor.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

namespace gtsam {
using symbol_shorthand::B;
using symbol_shorthand::P;

TEST(PlanarGyroFactor, evaluateError) {
  const double arw = 0.01;
  const double trueOmega = M_PI / 10.0;
  const double B1 = 0.3;
  // Measurement includes bias.
  double measuredOmega = trueOmega + B1;
  double deltaT = 1.0;

  PlanarGyroFactor factor(
      P(1), P(2), B(1),
      PlanarGyroMeasurement::fromRate(arw, measuredOmega, deltaT));

  const double initialRotation = M_PI / 4.0;
  Pose2 P1(0.0, 0.0, initialRotation);
  double error = 0.1;
  Pose2 P2(0.0, 0.0, initialRotation + trueOmega * deltaT - error);

  EXPECT(assert_equal(Vector3(0, 0, error), factor.evaluateError(P1, P2, B1),
                      1e-6))
}

TEST(PlanarGyroFactor, optimize) {
  using noiseModel::Diagonal;

  NonlinearFactorGraph graph;

  // Starting pose is known.
  graph.add(PriorFactor<Pose2>(P(0), Pose2(),
                               Diagonal::Sigmas(Vector3(0.001, 0.001, 0.001))));

  // BetweenFactors that simulate odometry.
  Pose2 p0 = Pose2(0, 0, 0);
  Pose2 p1 = Pose2(0, 0, 0.1);
  Pose2 p2 = Pose2(0.1, 0, 0.2);
  Pose2 p3 = Pose2(0.2, 0, 0.3);
  Pose2 p4 = Pose2(0.3, 0, 0.4);
  // When motionless, the rotation is known.
  // This is how we learn the bias.
  SharedDiagonal lowRotationNoise =
      noiseModel::Diagonal::Sigmas(Vector3(1e-3, 1e-3, 1e-3));
  graph.add(BetweenFactor<Pose2>(P(0), P(1), p0.between(p1), lowRotationNoise));

  // When moving, rotation is much less certain.
  SharedDiagonal highRotationNoise =
      noiseModel::Diagonal::Sigmas(Vector3(1e-3, 1e-3, 1));
  graph.add(
      BetweenFactor<Pose2>(P(1), P(2), p1.between(p2), highRotationNoise));
  graph.add(
      BetweenFactor<Pose2>(P(2), P(3), p2.between(p3), highRotationNoise));
  graph.add(
      BetweenFactor<Pose2>(P(3), P(4), p3.between(p4), highRotationNoise));

  // Bias prior: very uncertain.
  graph.add(PriorFactor<double>(B(0), 1.0, Diagonal::Sigmas(Vector1(1))));

  // Bias evolution.  Bias stability is an important parameter.
  SharedDiagonal biasNoise = Diagonal::Sigmas(Vector1(3e-4));
  graph.add(BetweenFactor<double>(B(0), B(1), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(1), B(2), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(2), B(3), 0, biasNoise));
  graph.add(BetweenFactor<double>(B(3), B(4), 0, biasNoise));

  // Gyro measurements affect rotation only.
  double arw = 1e-8;
  const double trueOmega = 0.1;
  const double bias = 1;  // large !
  const double measuredOmega = trueOmega + bias;
  double dt = 1.0;

  graph.add(PlanarGyroFactor(
      P(0), P(1), B(0),
      PlanarGyroMeasurement::fromRate(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(
      P(1), P(2), B(1),
      PlanarGyroMeasurement::fromRate(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(
      P(2), P(3), B(2),
      PlanarGyroMeasurement::fromRate(arw, measuredOmega, dt)));
  graph.add(PlanarGyroFactor(
      P(3), P(4), B(3),
      PlanarGyroMeasurement::fromRate(arw, measuredOmega, dt)));

  // Initial values should not matter.
  Values values;
  values.insert(B(0), 0.0);
  values.insert(B(1), 0.0);
  values.insert(B(2), 0.0);
  values.insert(B(3), 0.0);
  values.insert(B(4), 0.0);
  values.insert(P(0), Pose2());
  values.insert(P(1), Pose2());
  values.insert(P(2), Pose2());
  values.insert(P(3), Pose2());
  values.insert(P(4), Pose2());

  LevenbergMarquardtParams params;
  LevenbergMarquardtOptimizer optimizer(graph, values, params);
  Values result = optimizer.optimize();

  // Rotation increments are exactly what the "between" factor said.
  EXPECT(assert_equal(Pose2(0.0, 0.0, 0.0), result.at<Pose2>(P(0)), 1e-6));
  EXPECT(assert_equal(Pose2(0.0, 0.0, 0.1), result.at<Pose2>(P(1)), 1e-6));
  EXPECT(assert_equal(Pose2(0.1, 0.0, 0.2), result.at<Pose2>(P(2)), 1e-6));
  EXPECT(assert_equal(Pose2(0.2, 0.0, 0.3), result.at<Pose2>(P(3)), 1e-6));
  EXPECT(assert_equal(Pose2(0.3, 0.0, 0.4), result.at<Pose2>(P(4)), 1e-6));

  // Bias is correctly learned.
  EXPECT(assert_equal(1.0, result.at<double>(B(0)), 1e-6));
  EXPECT(assert_equal(1.0, result.at<double>(B(1)), 1e-6));
  EXPECT(assert_equal(1.0, result.at<double>(B(2)), 1e-6));
  EXPECT(assert_equal(1.0, result.at<double>(B(3)), 1e-6));
  EXPECT(assert_equal(1.0, result.at<double>(B(4)), 1e-6));

  Marginals marginals(graph, result);

  // Look at std dev because it's not so tiny.
  EXPECT(assert_equal(
      Vector3(0.001000, 0.001000, 0.001000),
      Vector3(marginals.marginalCovariance(P(0)).diagonal().cwiseSqrt()), 1e-6))
  EXPECT(assert_equal(
      Vector3(0.001414, 0.001414, 0.001414),
      Vector3(marginals.marginalCovariance(P(1)).diagonal().cwiseSqrt()), 1e-6))
  EXPECT(assert_equal(
      Vector3(0.001732, 0.001738, 0.002261),
      Vector3(marginals.marginalCovariance(P(2)).diagonal().cwiseSqrt()), 1e-6))
  EXPECT(assert_equal(
      Vector3(0.002003, 0.002030, 0.003242),
      Vector3(marginals.marginalCovariance(P(3)).diagonal().cwiseSqrt()), 1e-6))
  EXPECT(assert_equal(
      Vector3(0.002252, 0.002322, 0.004287),
      Vector3(marginals.marginalCovariance(P(4)).diagonal().cwiseSqrt()), 1e-6))

  // Bias variance is roughly constant.
  EXPECT(assert_equal(0.001005, sqrt(marginals.marginalCovariance(B(0))(0, 0)),
                      1e-6))
  EXPECT(assert_equal(0.001049, sqrt(marginals.marginalCovariance(B(1))(0, 0)),
                      1e-6))
  EXPECT(assert_equal(0.001091, sqrt(marginals.marginalCovariance(B(2))(0, 0)),
                      1e-6))
  EXPECT(assert_equal(0.001131, sqrt(marginals.marginalCovariance(B(3))(0, 0)),
                      1e-6))
  EXPECT(assert_equal(0.001170, sqrt(marginals.marginalCovariance(B(4))(0, 0)),
                      1e-6))
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
