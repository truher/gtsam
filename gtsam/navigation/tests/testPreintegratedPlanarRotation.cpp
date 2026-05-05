/**
 * @file   testPreintegratedPlanarRotation.cpp
 * @brief  Unit test for PreintegratedPlanarRotation
 * @author joel@truher.org
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/PreintegratedPlanarRotation.h>

#include <memory>

#include "gtsam/base/Matrix.h"
#include "gtsam/base/Vector.h"

using namespace gtsam;

//******************************************************************************
// Example where gyro measures small rotation about x-axis, with bias.
namespace biased_x_rotation {
const double omega = 0.1;
const double trueOmega = omega;
const double bias(1);
const double measuredOmega = trueOmega + bias;
const double deltaT = 0.5;
}  // namespace biased_x_rotation

//******************************************************************************
TEST(PreintegratedPlanarRotation, integrateGyroMeasurement) {
  // Example where IMU is identical to body frame, then omega is roll
  using namespace biased_x_rotation;

  // Check the value.
  Matrix1 H_bias;
  const internal::IncrementalPlanarRotation f{measuredOmega, deltaT};
  const Rot2 incrR = f(bias, H_bias);
  const Rot2 expected = Rot2(omega * deltaT);
  EXPECT(assert_equal(expected, incrR, 1e-9))

  // Check the derivative:
  EXPECT(assert_equal(numericalDerivative11<Rot2, double>(f, bias), H_bias))

  // Check value of deltaRij() after integration.
  PreintegratedPlanarRotation pim(1);
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  
  Matrix1 F = I_1x1;

  EXPECT(assert_equal(expected, pim.deltaRij(), 1e-9))

  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaRij().inverse().AdjointMap(), F))

  // Make sure delRdelBiasOmega is H_bias after integration.
  EXPECT(assert_equal<Matrix1>(H_bias, pim.delRdelBiasOmega()))

  // Check if we make a correction to the bias, the value and Jacobian are
  // correct. Note that the bias is subtracted from the measurement, and the
  // integration time is taken into account, so we expect -deltaT*delta change.
  Matrix1 H;
  const double delta = 0.05;
  const double biasOmegaIncr = delta;
  Rot2 corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  EQUALITY(Vector1(-deltaT * delta), expected.logmap(corrected))
  EXPECT(assert_equal(Rot2((omega - delta) * deltaT), corrected, 1e-9))

  // Check the derivative matches the numerical one
  auto g = [&](const double& increment) {
    return pim.biascorrectedDeltaRij(increment, {});
  };
  Matrix1 expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));
  
  // Let's integrate a second IMU measurement and check the Jacobian update:
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  EXPECT(assert_equal(expectedH, H));
}

//******************************************************************************


TEST(PreintegratedPlanarRotation, integrateGyroMeasurementWithTransform) {
  // Example where IMU is rotated, so measured omega indicates pitch.
  using namespace biased_x_rotation;

  // Check the value.
  Matrix1 H_bias;
  const internal::IncrementalPlanarRotation f{measuredOmega, deltaT};
  const Rot2 expected = Rot2(omega * deltaT);
  EXPECT(assert_equal(expected, f(bias, H_bias), 1e-9))

  // Check the derivative:
  EXPECT(assert_equal(numericalDerivative11<Rot2, double>(f, bias), H_bias))

  // Check value of deltaRij() after integration.
  PreintegratedPlanarRotation pim(1);
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);

  Matrix1 F = I_1x1;

  EXPECT(assert_equal(expected, pim.deltaRij(), 1e-9))

  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaRij().inverse().AdjointMap(), F))

  // Make sure delRdelBiasOmega is H_bias after integration.
  EXPECT(assert_equal<Matrix1>(H_bias, pim.delRdelBiasOmega()))

  // Check the bias correction in same way, but will now yield pitch change.
  Matrix1 H;
  const double delta = 0.05;
  const double biasOmegaIncr = delta;
  Rot2 corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  EQUALITY(Vector1(-deltaT * delta), expected.logmap(corrected))
  EXPECT(assert_equal(Rot2((omega - delta) * deltaT), corrected, 1e-9))

  // Check the derivative matches the *expectedH* one
  auto g = [&](const double& increment) {
    return pim.biascorrectedDeltaRij(increment, {});
  };
  Matrix1 expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));

  // Let's integrate a second IMU measurement and check the Jacobian update:
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));
}

TEST(PreintegratedPlanarRotation, integrateGyroMeasurementWithArbitraryTransform) {
  // Example with a non-axis-aligned transform and some position.
  using namespace biased_x_rotation;

  // Check the derivative:
  Matrix1 H_bias;
  const internal::IncrementalPlanarRotation f{measuredOmega, deltaT};
  f(bias, H_bias);
  EXPECT(assert_equal(numericalDerivative11<Rot2, double>(f, bias), H_bias))

  // Check derivative of deltaRij() after integration.
  PreintegratedPlanarRotation pim(1);
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);

  Matrix1 F = I_1x1;
 
  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaRij().inverse().AdjointMap(), F))

  // Make sure delRdelBiasOmega is H_bias after integration.
  EXPECT(assert_equal<Matrix1>(H_bias, pim.delRdelBiasOmega()))

  // Check the bias correction in same way, but will now yield pitch change.
  Matrix1 H;
  const double delta = 0.05;
  const double biasOmegaIncr = delta;
  Rot2 corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);

  // Check the derivative matches the numerical one
  auto g = [&](const double& increment) {
    return pim.biascorrectedDeltaRij(increment, {});
  };
  Matrix1 expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));

  // Let's integrate a second IMU measurement and check the Jacobian update:
  pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
