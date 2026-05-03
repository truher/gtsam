/**
 * @file testPlanarAHRSFactor.cpp
 * @date May 1, 2026
 * @author joel@truher.org
 * @brief tests for PlanarAHRSFactor
 */


#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/debug.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/navigation/PlanarAHRSFactor.h>
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
using namespace gtsam;

// Convenience for named keys
using symbol_shorthand::B;
using symbol_shorthand::R;

// Define covariance matrices
double gyroNoiseVar = 0.01;
const Matrix1 kMeasuredOmegaCovariance = gyroNoiseVar * I_1x1;

//******************************************************************************
namespace {
PreintegratedPlanarAhrsMeasurements integrateMeasurements(
    const Vector1& biasHat, const list<Vector1>& measuredOmegas,
    const list<double>& deltaTs) {
  PreintegratedPlanarAhrsMeasurements result(biasHat, I_1x1);

  list<Vector1>::const_iterator itOmega = measuredOmegas.begin();
  list<double>::const_iterator itDeltaT = deltaTs.begin();
  for (; itOmega != measuredOmegas.end(); ++itOmega, ++itDeltaT) {
    result.integrateMeasurement(*itOmega, *itDeltaT);
  }

  return result;
}
}  // namespace

//******************************************************************************
TEST(PlanarAHRSFactor, PreintegratedPlanarAhrsMeasurements) {
  // Linearization point
  Vector1 biasHat(0);  ///< Current estimate of angular rate bias

  // Measurements
  Vector1 measuredOmega(M_PI / 100.0);
  double deltaT = 0.5;

  // Expected preintegrated values
  Rot2 expectedDeltaR1 = Rot2(0.5 * M_PI / 100.0);

  // Actual preintegrated values
  PreintegratedPlanarAhrsMeasurements actual1(biasHat, kMeasuredOmegaCovariance);
  actual1.integrateMeasurement(measuredOmega, deltaT);

  EXPECT(assert_equal(expectedDeltaR1, Rot2(actual1.deltaRij()), 1e-6));
  DOUBLES_EQUAL(deltaT, actual1.deltaTij(), 1e-6);

  // Check the covariance
  Matrix1 expectedMeasCov = kMeasuredOmegaCovariance * deltaT;
  EXPECT(assert_equal(expectedMeasCov, actual1.preintMeasCov(), 1e-6));

  // Integrate again
  Rot2 expectedDeltaR2 = Rot2(2.0 * 0.5 * M_PI / 100.0);

  // Actual preintegrated values
  PreintegratedPlanarAhrsMeasurements actual2 = actual1;
  actual2.integrateMeasurement(measuredOmega, deltaT);

  EXPECT(assert_equal(expectedDeltaR2, Rot2(actual2.deltaRij()), 1e-6));
  DOUBLES_EQUAL(deltaT * 2, actual2.deltaTij(), 1e-6);
}

//******************************************************************************
TEST(PlanarAHRSFactor, PreintegratedPlanarAhrsMeasurementsConstructor) {
  Matrix1 gyroscopeCovariance = I_1x1 * 0.4;
  PreintegratedPlanarRotationParams params(gyroscopeCovariance);
  Vector1 bias(1.0);  ///< Current estimate of angular rate bias
  Rot2 deltaRij(Rot2(M_PI / 12.0));
  double deltaTij = 0.02;
  Matrix1 delRdelBiasOmega = I_1x1 * 0.5;
  Matrix1 preintMeasCov = I_1x1 * 0.2;
  PreintegratedPlanarAhrsMeasurements actualPim(
      std::make_shared<PreintegratedPlanarRotationParams>(params), bias, deltaTij,
      deltaRij, delRdelBiasOmega, preintMeasCov);
  EXPECT(assert_equal(gyroscopeCovariance,
                      actualPim.p().getGyroscopeCovariance(), 1e-6));
  EXPECT(assert_equal(bias, actualPim.biasHat(), 1e-6));
  DOUBLES_EQUAL(deltaTij, actualPim.deltaTij(), 1e-6);
  EXPECT(assert_equal(deltaRij, Rot2(actualPim.deltaRij()), 1e-6));
  EXPECT(assert_equal(delRdelBiasOmega, actualPim.delRdelBiasOmega(), 1e-6));
  EXPECT(assert_equal(preintMeasCov, actualPim.preintMeasCov(), 1e-6));
}

/* ************************************************************************* */
TEST(PlanarAHRSFactor, PIMPredict) {
  // Modernized version of predictTest, calling predict on the PIM directly.
  Vector1 bias(0);

  // Measurements
  Vector1 measuredOmega(M_PI / 10.0);
  double deltaT = 0.2;
  PreintegratedPlanarAhrsMeasurements pim(bias, kMeasuredOmegaCovariance);
  for (int i = 0; i < 1000; ++i) {
    pim.integrateMeasurement(measuredOmega, deltaT);
  }

  // Predict
  Rot2 Ri;
  Rot2 expectedRot = Rot2(20 * M_PI);
  // The new predict method lives on the PIM object
  Rot2 actualRot = pim.predict(Ri, bias);
  EXPECT(assert_equal(expectedRot, actualRot, 1e-6));
}

/* ************************************************************************* */
TEST(PlanarAHRSFactor, PIMComputeError) {
  // Tests the modernized computeError and its Jacobians, now on the PIM.
  Vector1 bias(0.1);
  Rot2 Ri(Rot2(M_PI / 12.0));
  Rot2 Rj(Rot2(M_PI / 12.0 + M_PI / 100.0));

  // Measurements
  Vector1 measuredOmega(M_PI / 100 + 0.1);
  double deltaT = 1.0;
  PreintegratedPlanarAhrsMeasurements pim(Vector1(0), kMeasuredOmegaCovariance);
  pim.integrateMeasurement(measuredOmega, deltaT);

  // Use a wrapper to call the new PIM::computeError for numerical derivatives
  auto f = [&pim](const Rot2& r1, const Rot2& r2, const Vector1& b) -> Vector1 {
    return pim.computeError(r1, r2, b);
  };

  // Calculate analytical Jacobians
  Matrix1 H1, H2, H3;
  (void)pim.computeError(Ri, Rj, bias, H1, H2, H3);

  // Calculate numerical Jacobians
  Matrix1 H1_numerical = numericalDerivative31(f, Ri, Rj, bias);
  Matrix1 H2_numerical = numericalDerivative32(f, Ri, Rj, bias);
  Matrix1 H3_numerical = numericalDerivative33(f, Ri, Rj, bias);

  // Compare
  EXPECT(assert_equal(H1_numerical, H1, 1e-6));
  EXPECT(assert_equal(H2_numerical, H2, 1e-6));
  EXPECT(assert_equal(H3_numerical, H3, 1e-6));
}

/* ************************************************************************* */
TEST(PlanarAHRSFactor, Error) {
  // Linearization point
  Vector1 bias(0.);  // Bias
  Rot2 Ri(Rot2(M_PI / 12.0));
  Rot2 Rj(Rot2(M_PI / 12.0 + M_PI / 100.0));

  // Measurements
  Vector1 measuredOmega(M_PI / 100);
  double deltaT = 1.0;
  PreintegratedPlanarAhrsMeasurements pim(bias, kMeasuredOmegaCovariance);
  pim.integrateMeasurement(measuredOmega, deltaT);

  // Create factor
  PlanarAHRSFactor factor(R(1), R(2), B(1), pim);

  // Check value
  Vector1 errorActual = factor.evaluateError(Ri, Rj, bias);
  Vector1 errorExpected(0);
  EXPECT(assert_equal(Vector(errorExpected), Vector(errorActual), 1e-6));

  // Check Derivatives
  Values values;
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

/* ************************************************************************* */
TEST(PlanarAHRSFactor, ErrorWithBiases) {
  // Linearization point
  Vector1 bias(0.3);
  Rot2 Ri(Rot2::Expmap(Vector1(M_PI / 4.0)));
  Rot2 Rj(Rot2::Expmap(Vector1(M_PI / 4.0 + M_PI / 10.0)));

  // Measurements
  Vector1 measuredOmega(M_PI / 10.0 + 0.3);
  double deltaT = 1.0;
  PreintegratedPlanarAhrsMeasurements pim(Vector1(0), kMeasuredOmegaCovariance);
  pim.integrateMeasurement(measuredOmega, deltaT);

  // Create factor
  PlanarAHRSFactor factor(R(1), R(2), B(1), pim);

  // Check value
  Vector1 errorExpected(0);
  Vector1 errorActual = factor.evaluateError(Ri, Rj, bias);
  EXPECT(assert_equal(errorExpected, errorActual, 1e-6));

  // Check Derivatives
  Values values;
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

//******************************************************************************
TEST(PlanarAHRSFactor, PartialDerivativeExpmap) {
  // Linearization point
  Vector1 biasOmega(0);

  // Measurements
  Vector1 measuredOmega(0.1);
  double deltaT = 0.5;

  auto f = [&](const Vector1& biasOmega) {
    return Rot2::Expmap((measuredOmega - biasOmega) * deltaT);
  };

  // Compute numerical derivatives
  Matrix expectedH = numericalDerivative11<Rot2, Vector1>(f, biasOmega);

  const Matrix1 Jr =
      Rot2::ExpmapDerivative((measuredOmega - biasOmega) * deltaT);

  Matrix1 actualH = -Jr * deltaT;  // the delta bias appears with the minus sign

  // Compare Jacobians
  EXPECT(assert_equal(expectedH, actualH, 1e-3));
  // 1e-3 needs to be added only when using quaternions for rotations
}

//******************************************************************************
TEST(PlanarAHRSFactor, PartialDerivativeLogmap) {
  // Linearization point
  Vector1 thetaHat(0.1);  ///< Current estimate of rotation rate bias

  auto f = [thetaHat](const Vector1 deltaTheta) {
    return Rot2::Logmap(
        Rot2::Expmap(thetaHat).compose(Rot2::Expmap(deltaTheta)));
  };

  // Compute numerical derivatives
  Vector1 deltaTheta(0);
  Matrix expectedH = numericalDerivative11<Vector1, Vector1>(f, deltaTheta);

  const Vector1 x = thetaHat;          // parametrization of so(3)
  const Matrix1 X = Matrix1(0);  // element of Lie algebra so(3): X = x^
  double norm = x.norm();
  const Matrix1 actualH =
      I_1x1 + 0.5 * X +
      (1 / (norm * norm) - (1 + cos(norm)) / (2 * norm * sin(norm))) * X * X;

  // Compare Jacobians
  EXPECT(assert_equal(expectedH, actualH));
}

//******************************************************************************
TEST(PlanarAHRSFactor, fistOrderExponential) {
  // Linearization point
  Vector1 biasOmega(0);

  // Measurements
  Vector1 measuredOmega(0.1);
  double deltaT = 1.0;

  // change w.r.t. linearization point
  double alpha = 0.0;
  Vector1 deltaBiasOmega(alpha);

  const Matrix1 Jr =
      Rot2::ExpmapDerivative((measuredOmega - biasOmega) * deltaT);

  Matrix1 delRdelBiasOmega =
      -Jr * deltaT;  // the delta bias appears with the minus sign

  const Matrix expectedRot =
      Rot2::Expmap((measuredOmega - biasOmega - deltaBiasOmega) * deltaT)
          .matrix();

  const Matrix2 hatRot =
      Rot2::Expmap((measuredOmega - biasOmega) * deltaT).matrix();
  const Matrix2 actualRot =
      hatRot * Rot2::Expmap(delRdelBiasOmega * deltaBiasOmega).matrix();

  // Compare Jacobians
  EXPECT(assert_equal(expectedRot, actualRot));
}

//******************************************************************************
TEST(PlanarAHRSFactor, FirstOrderPreIntegratedPlanarMeasurements) {
  // Linearization point
  Vector1 bias = Vector1::Zero();  ///< Current estimate of rotation rate bias

  // Measurements
  list<Vector1> measuredOmegas;
  list<double> deltaTs;
  measuredOmegas.push_back(Vector1(M_PI / 100.0));
  deltaTs.push_back(0.01);
  measuredOmegas.push_back(Vector1(M_PI / 100.0));
  deltaTs.push_back(0.01);
  for (int i = 1; i < 100; i++) {
    measuredOmegas.push_back(
        Vector1(M_PI / 100.0));
    deltaTs.push_back(0.01);
  }

  // Actual preintegrated values
  PreintegratedPlanarAhrsMeasurements preintegrated =
      integrateMeasurements(bias, measuredOmegas, deltaTs);

  auto f = [&](const Vector1& bias) {
    return integrateMeasurements(bias, measuredOmegas, deltaTs).deltaRij();
  };

  // Compute numerical derivatives
  Matrix expectedDelRdelBias = numericalDerivative11<Rot2, Vector1>(f, bias);
  Matrix expectedDelRdelBiasOmega = expectedDelRdelBias;

  // should be around -1, so expected is correct, actual is wrong.
  // Compare Jacobians
  EXPECT(assert_equal(expectedDelRdelBiasOmega,
                      preintegrated.delRdelBiasOmega(), 1e-3));
  // 1e-3 needs to be added only when using quaternions for rotations
}

//******************************************************************************
TEST(PlanarAHRSFactor, ErrorWithBiasesAndSensorBodyDisplacement) {
  Vector1 bias(0.3);
  Rot2 Ri(Rot2::Expmap(Vector1(M_PI / 4.0)));
  Rot2 Rj(Rot2::Expmap(Vector1(M_PI / 4.0 + M_PI / 10.0)));

  // Measurements
  Vector1 measuredOmega(M_PI / 10.0 + 0.3);
  double deltaT = 1.0;

  auto p = std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>();
  p->gyroscopeCovariance = kMeasuredOmegaCovariance;
  PreintegratedPlanarAhrsMeasurements pim(p, Vector1::Zero());

  pim.integrateMeasurement(measuredOmega, deltaT);

  // Check preintegrated covariance
  EXPECT(assert_equal(kMeasuredOmegaCovariance, pim.preintMeasCov()));

  // Create factor
  PlanarAHRSFactor factor(R(1), R(2), B(1), pim);

  // Check Derivatives
  Values values;
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-6);
}

//******************************************************************************
TEST(PlanarAHRSFactor, PIM_predict_and_Jacobians) {
  // --- Setup ---
  Vector1 bias(0.1);
  Rot2 Ri = Rot2(M_PI / 4.0);
  auto p = std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>();
  p->gyroscopeCovariance = kMeasuredOmegaCovariance;

  PreintegratedPlanarAhrsMeasurements pim(p, Vector1::Zero());

  // Integrate a few measurements
  Vector1 measuredOmega(M_PI / 10.0);
  double deltaT = 0.5;
  pim.integrateMeasurement(measuredOmega, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);

  // --- Test Prediction Value ---
  // Call the new predict method without requesting Jacobians
  Rot2 predictedRot = pim.predict(Ri, bias, {}, {});

  // Calculate expected value manually for verification
  Vector1 biasOmegaIncr = bias - pim.biasHat();
  Rot2 expected_biascorrected_delta = pim.biascorrectedDeltaRij(biasOmegaIncr);
  Rot2 expectedRot = Ri.compose(expected_biascorrected_delta);
  EXPECT(assert_equal(expectedRot, predictedRot, 1e-6));

  // --- Test Jacobians ---
  // Define a wrapper for numerical derivatives
  auto f = [&pim](const Rot2& r, const Vector1& b) { return pim.predict(r, b); };

  // Get analytical Jacobians from the predict call
  Matrix1 H1_actual, H2_actual;
  (void)pim.predict(Ri, bias, H1_actual, H2_actual);

  // Get numerical Jacobians
  Matrix1 H1_numerical = numericalDerivative21(f, Ri, bias);
  Matrix1 H2_numerical = numericalDerivative22(f, Ri, bias);

  // Compare analytical and numerical Jacobians
  EXPECT(assert_equal(H1_numerical, H1_actual, 1e-7));
  EXPECT(assert_equal(H2_numerical, H2_actual, 1e-7));
}

//******************************************************************************
// Test predict with Coriolis enabled
TEST(PlanarAHRSFactor, PIM_predict_and_Jacobians_with_Coriolis) {
  // --- Setup ---
  Vector1 bias(0.1);
  Rot2 Ri = Rot2(M_PI / 4.0);

  auto p = std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>();
  p->gyroscopeCovariance = kMeasuredOmegaCovariance;

  PreintegratedPlanarAhrsMeasurements pim(p, Vector1::Zero());

  // Integrate a few measurements
  Vector1 measuredOmega(0.1);
  double deltaT = 0.5;
  pim.integrateMeasurement(measuredOmega, deltaT);

  // --- Test Jacobians ---
  // Define a wrapper for numerical derivatives
  auto f = [&pim](const Rot2& r, const Vector1& b) { return pim.predict(r, b); };

  // Get analytical Jacobians from the predict call
  Matrix1 H1_actual, H2_actual;
  (void)pim.predict(Ri, bias, H1_actual, H2_actual);

  // Get numerical Jacobians
  Matrix1 H1_numerical = numericalDerivative21(f, Ri, bias);
  Matrix1 H2_numerical = numericalDerivative22(f, Ri, bias);

  // Compare analytical and numerical Jacobians
  EXPECT(assert_equal(H1_numerical, H1_actual, 1e-7));
  EXPECT(assert_equal(H2_numerical, H2_actual, 1e-7));
}
//******************************************************************************
TEST(PlanarAHRSFactor, graphTest) {
  // linearization point
  Rot2 Ri(Rot2(0));
  Rot2 Rj(Rot2(0));
  Vector1 bias(0);

  // PreIntegrator
  Vector1 biasHat(0);
  PreintegratedPlanarAhrsMeasurements pim(biasHat, kMeasuredOmegaCovariance);

  // Pre-integrate measurements
  Vector1 measuredOmega(M_PI / 20);
  double deltaT = 1;

  // Create Factor
  noiseModel::Base::shared_ptr model =  //
      noiseModel::Gaussian::Covariance(pim.preintMeasCov());
  NonlinearFactorGraph graph;
  Values values;
  for (size_t i = 0; i < 5; ++i) {
    pim.integrateMeasurement(measuredOmega, deltaT);
  }

  // pim.print("Pre integrated measurements");
  PlanarAHRSFactor factor(R(1), R(2), B(1), pim);
  values.insert(R(1), Ri);
  values.insert(R(2), Rj);
  values.insert(B(1), bias);
  graph.push_back(factor);
  LevenbergMarquardtOptimizer optimizer(graph, values);
  Values result = optimizer.optimize();
  Rot2 expectedRot(Rot2(M_PI / 4));
  EXPECT(assert_equal(expectedRot, result.at<Rot2>(R(2))));
}

/* ************************************************************************* */
TEST(PlanarAHRSFactor, bodyPSensorWithBias) {
  using noiseModel::Diagonal;

  int numRotations = 10;
  const Vector1 noiseBetweenBiasSigma(3.0e-6);
  SharedDiagonal biasNoiseModel = Diagonal::Sigmas(noiseBetweenBiasSigma);

  // Measurements in the sensor frame:
  const double omega = 0.1;
  const Vector1 realOmega(omega);
  const Vector1 realBias(1);  // large !
  const Vector1 measuredOmega = realOmega + realBias;

  auto p = std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>();
  p->gyroscopeCovariance = 1e-8 * I_1x1;
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
  // that this bias is constant over time In theory, after optimization, we
  // should recover that same bias estimate
  graph.addPrior(B(0), realBias, priorNoiseBias);
  values.insert(B(0), realBias);

  // Now add IMU factors and bias noise models
  const Vector1 zeroBias(0);
  for (int i = 1; i < numRotations; i++) {
    PreintegratedPlanarAhrsMeasurements pim(p, realBias);
    for (int j = 0; j < 200; ++j)
      pim.integrateMeasurement(measuredOmega, deltaT);

    // Create factors
    graph.emplace_shared<PlanarAHRSFactor>(R(i - 1), R(i), B(i - 1), pim);
    graph.emplace_shared<BetweenFactor<Vector1> >(B(i - 1), B(i), zeroBias,
                                                  biasNoiseModel);

    values.insert(R(i), Rot2());
    values.insert(B(i), realBias);
  }

  // Finally, optimize, and get bias at last time step
  LevenbergMarquardtParams params;
  // params.setVerbosityLM("SUMMARY");
  Values result = LevenbergMarquardtOptimizer(graph, values, params).optimize();
  const Vector1 biasActual = result.at<Vector1>(B(numRotations - 1));

  // Bias should be a self-fulfilling prophesy:
  EXPECT(assert_equal(realBias, biasActual, 1e-3));

  // Check that the successive rotations are all `omega` apart:
  for (int i = 0; i < numRotations; i++) {
    Rot2 expectedRot = Rot2(omega * i);
    Rot2 actualRot = result.at<Rot2>(R(i));
    EXPECT(assert_equal(expectedRot, actualRot, 1e-3));
  }
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
