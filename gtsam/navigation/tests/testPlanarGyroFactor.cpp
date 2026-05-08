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
  const double arw = 0.01;
  PlanarGyroMeasurement measurement(arw);

  const double trueOmega = M_PI / 10.0;
  const double bias = 0.3;

  // Measurement includes bias.
  double measuredOmega = trueOmega + bias;
  double deltaT = 1.0;
  measurement.integrate(measuredOmega, deltaT);

  PlanarGyroFactor factor(R(1), R(2), B(1), measurement);

  // Estimate is correct.
  Rot2 Ri(M_PI / 4.0);
  Rot2 Rj(M_PI / 4.0 + M_PI / 10.0);
  DOUBLES_EQUAL(0, factor.evaluateError(Ri, Rj, bias)(0), 1e-6);

  Values values;
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}


TEST(PlanarGyroFactor, fistOrderExponential) {
  double biasOmega = 0;

  double measuredOmega = 0.1;
  double deltaT = 1.0;

  // change w.r.t. linearization point
  double alpha = 0.0;
  double deltaBiasOmega = alpha;

  double delRdelBiasOmega =
      -1.0 * deltaT;  // the delta bias appears with the minus sign

  const Matrix expectedRot =
      Rot2::fromAngle((measuredOmega - biasOmega - deltaBiasOmega) * deltaT)
          .matrix();

  const Matrix2 hatRot =
      Rot2::fromAngle((measuredOmega - biasOmega) * deltaT).matrix();
  const Matrix2 actualRot =
      hatRot * Rot2::fromAngle(delRdelBiasOmega * deltaBiasOmega).matrix();

  EXPECT(assert_equal(expectedRot, actualRot));
}




TEST(PlanarGyroFactor, PIM_predict_and_Jacobians) {
  const double arw = 0.01;
  double bias = 0.1;
  Rot2 Ri = Rot2(M_PI / 4.0);

  PlanarGyroMeasurement measurement(arw);

  double measuredOmega = M_PI / 10.0;
  double deltaT = 0.5;
  measurement.integrate(measuredOmega, deltaT);
  measurement.integrate(measuredOmega, deltaT);

  Rot2 predictedRot = measurement.predict(Ri, bias, {}, {});

  // Calculate expected value manually for verification
  double biasOmegaIncr = bias;
  Rot2 expected_biascorrected_delta = measurement.deltaR(biasOmegaIncr);
  Rot2 expectedRot = Ri.compose(expected_biascorrected_delta);
  EXPECT(assert_equal(expectedRot, predictedRot, 1e-6));

  auto f = [&measurement](const Rot2& r, const double& b) { 
    return measurement.predict(r, b); 
  };

  Matrix1 H1_actual, H2_actual;
  (void)measurement.predict(Ri, bias, H1_actual, H2_actual);


  Matrix1 H1_numerical = numericalDerivative21(f, Ri, bias);
  Matrix1 H2_numerical = numericalDerivative22(f, Ri, bias);

  EXPECT(assert_equal(H1_numerical, H1_actual, 1e-7));
  EXPECT(assert_equal(H2_numerical, H2_actual, 1e-7));
}


TEST(PlanarGyroFactor, graphTest) {
  const double arw = 0.01;
  Rot2 Ri(Rot2(0));
  Rot2 Rj(Rot2(M_PI / 4));
  double bias = 0;

  PlanarGyroMeasurement measurement(arw);

  // Pre-integrate measurements
  double measuredOmega = M_PI / 20;
  double deltaT = 1;

  // Create Factor
  noiseModel::Base::shared_ptr model =  //
      noiseModel::Gaussian::Covariance(measurement.variance());
  NonlinearFactorGraph graph;
  Values values;
  for (size_t i = 0; i < 5; ++i) {
    measurement.integrate(measuredOmega, deltaT);
  }

  PlanarGyroFactor factor(R(1), R(2), B(1), measurement);
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  graph.push_back(factor);
  LevenbergMarquardtOptimizer optimizer(graph, values);
  Values result = optimizer.optimize();
  Rot2 expectedRot(Rot2(M_PI / 4));
  EXPECT(assert_equal(expectedRot, result.at<Rot2>(R(2))));
}

TEST(PlanarGyroFactor, bodyPSensorWithBias) {
  using noiseModel::Diagonal;

  int numRotations = 10;
  const Vector1 noiseBetweenBiasSigma(3.0e-6);
  SharedDiagonal biasNoiseModel = Diagonal::Sigmas(noiseBetweenBiasSigma);

  // Measurements in the sensor frame:
  const double omega = 0.1;
  const double realOmega = omega;
  const double realBias = 1;  // large !
  const double measuredOmega = realOmega + realBias;

  double deltaT = 0.005;

  // Specify noise values on priors
  const Vector1 priorNoisePoseSigmas(0.001);
  const Vector1 priorNoiseBiasSigmas(0.5e-1);
  SharedDiagonal priorNoisePose = Diagonal::Sigmas(priorNoisePoseSigmas);
  SharedDiagonal priorNoiseBias = Diagonal::Sigmas(priorNoiseBiasSigmas);

  // Create a factor graph with priors on initial pose, velocity and bias
  NonlinearFactorGraph graph;
  Values values;

  graph.addPrior(R(0), Rot2(), priorNoisePose);
  values.insert(R(0), Rot2());

  // The key to this test is that we specify the bias, in the sensor frame, as
  // known a priori. We also create factors below that encode our assumption
  // that this bias is constant over time. In theory, after optimization, we
  // should recover that same bias estimate
  graph.addPrior(B(0), realBias, priorNoiseBias);
  values.insert(B(0), realBias);

  // Now add IMU factors and bias noise models
  const double zeroBias = 0;
  for (int i = 1; i < numRotations; i++) {
    PlanarGyroMeasurement pim(1e-8);
    for (int j = 0; j < 200; ++j) pim.integrate(measuredOmega, deltaT);

    // Create factors
    graph.emplace_shared<PlanarGyroFactor>(R(i - 1), R(i), B(i - 1), pim);
    graph.emplace_shared<BetweenFactor<double> >(B(i - 1), B(i), zeroBias,
                                                 biasNoiseModel);

    values.insert(R(i), Rot2());
    values.insert(B(i), realBias);
  }

  // Finally, optimize, and get bias at last time step
  LevenbergMarquardtParams params;
  // params.setVerbosityLM("SUMMARY");
  // default is 1e-5, 1e-6 is required to pass
  params.setAbsoluteErrorTol(1e-6);
  Values result = LevenbergMarquardtOptimizer(graph, values, params).optimize();
  const double biasActual = result.at<double>(B(numRotations - 1));

  // Bias should be a self-fulfilling prophesy:
  EXPECT(assert_equal(realBias, biasActual, 1e-3));

  // Check that the successive rotations are all `omega` apart:
  for (int i = 0; i < numRotations; i++) {
    Rot2 expectedRot = Rot2(omega * i);
    Rot2 actualRot = result.at<Rot2>(R(i));
    EXPECT(assert_equal(expectedRot, actualRot, 1e-3));
  }
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
