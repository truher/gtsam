/**
 * @file   testPlanarGyro.cpp
 * @brief  Unit test for PlanarGyroMeasurement
 * @author joel@truher.org
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/PlanarGyroMeasurement.h>

#include <memory>

#include "gtsam/base/Matrix.h"
#include "gtsam/base/Vector.h"

using namespace gtsam;

// Example where gyro measures small rotation, with bias.
namespace biased_x_rotation {
const double omega = 0.1;
const double trueOmega = omega;
const double bias(1);
const double measuredOmega = trueOmega + bias;
const double deltaT = 0.5;
}  // namespace biased_x_rotation

TEST(PlanarGyroMeasurement, integrateGyroMeasurement) {
  // Example where IMU is identical to body frame, then omega is roll
  using namespace biased_x_rotation;

  const Rot2 expected = Rot2(omega * deltaT);

  // Check value of deltaRij() after integration.
  PlanarGyroMeasurement pim(1, bias);
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);
  
  Matrix1 F = I_1x1;

  EXPECT(assert_equal(expected, pim.deltaRij(), 1e-9))

  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaRij().inverse().AdjointMap(), F))

  // Make sure delRdelBiasOmega is H_bias after integration.
  Matrix1 H_bias = I_1x1 * -deltaT;
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
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);

  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  EXPECT(assert_equal(expectedH, H));
}


TEST(PlanarGyroMeasurement, integrateGyroMeasurementWithTransform) {
  // Example where IMU is rotated, so measured omega indicates pitch.
  using namespace biased_x_rotation;

  // Check the value.
  const Rot2 expected = Rot2(omega * deltaT);

  // Check value of deltaRij() after integration.
  PlanarGyroMeasurement pim(1, bias);
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);

  Matrix1 F = I_1x1;

  EXPECT(assert_equal(expected, pim.deltaRij(), 1e-9))

  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaRij().inverse().AdjointMap(), F))

  // Make sure delRdelBiasOmega is H_bias after integration.
  Matrix1 H_bias = I_1x1 * -deltaT;
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
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);
  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));
}

TEST(PlanarGyroMeasurement, integrateGyroMeasurementWithArbitraryTransform) {
  // Example with a non-axis-aligned transform and some position.
  using namespace biased_x_rotation;

  Matrix1 H_bias = I_1x1 * -deltaT;

  // Check derivative of deltaRij() after integration.
  PlanarGyroMeasurement pim(1, bias);
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);

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
  // pim.integrateGyroMeasurement(measuredOmega, bias, deltaT);
  pim.integrateMeasurement(measuredOmega, deltaT);

  corrected = pim.biascorrectedDeltaRij(biasOmegaIncr, H);
  expectedH = numericalDerivative11<Rot2, double>(g, biasOmegaIncr);
  EXPECT(assert_equal(expectedH, H));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
