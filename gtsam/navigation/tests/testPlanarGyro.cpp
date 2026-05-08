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

namespace gtsam {

TEST(PlanarGyroMeasurement, integrateGyroMeasurement) {
  const double omega = 0.1;
  const double deltaT = 0.5;
  const Rot2 expected = Rot2(omega * deltaT);
  PlanarGyroMeasurement pim(1);
  pim.integrateMeasurement(omega, deltaT);

  // Check integration.
  EXPECT(assert_equal(expected, pim.deltaR_, 1e-9))

  // Check that system matrix F is the first derivative of compose:
  EXPECT(assert_equal<Matrix1>(pim.deltaR_.inverse().AdjointMap(), I_1x1))

  // Check if we make a correction to the bias, the value and Jacobian are
  // correct. Note that the bias is subtracted from the measurement, and the
  // integration time is taken into account, so we expect -deltaT*delta change.
  const double delta = 0.05;
  Matrix1 H;
  Rot2 corrected = pim.biascorrectedDeltaR(delta, H);
  EQUALITY(Vector1(-deltaT * delta), expected.logmap(corrected))
  EXPECT(assert_equal(Rot2((omega - delta) * deltaT), corrected, 1e-9))

  // Check the derivative matches the numerical one.
  auto g = [&](const double& increment) {
    return pim.biascorrectedDeltaR(increment, {});
  };

  Matrix1 expectedH = numericalDerivative11<Rot2, double>(g, delta);
  EXPECT(assert_equal(expectedH, H));

  // Integrate a second IMU measurement.
  pim.integrateMeasurement(omega, deltaT);

  // Check the Jacobian update.
  expectedH = numericalDerivative11<Rot2, double>(g, delta);
  corrected = pim.biascorrectedDeltaR(delta, H);
  EXPECT(assert_equal(expectedH, H));
}
} // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
