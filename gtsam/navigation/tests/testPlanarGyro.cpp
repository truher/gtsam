/**
 * @file   testPlanarGyro.cpp
 * @brief  Unit test for PlanarGyroMeasurement
 * @author joel@truher.org
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/PlanarGyroMeasurement.h>

namespace gtsam {

TEST(PlanarGyroMeasurement, integrate) {
  const double arw = 1.0;
  PlanarGyroMeasurement measurement(arw);
  const double omega = 0.1;
  const double deltaT = 0.5;
  measurement.integrate(omega, deltaT);

  // Check integration.
  DOUBLES_EQUAL(0.05, measurement.deltaR_.theta(), 1e-9)

  // Check the effect of bias.
  const double bias = 0.05;
  Matrix1 H;
  Rot2 corrected = measurement.deltaR(bias, H);
  // (0.1 - 0.05) * 0.5 = 0.025
  DOUBLES_EQUAL(0.025, corrected.theta(), 1e-9)

  // Check that the derivative matches the numerical one.
  auto g = [&](const double& increment) {
    return measurement.deltaR(increment, {});
  };

  Matrix1 expectedH = numericalDerivative11<Rot2, double>(g, bias);
  EXPECT(assert_equal(expectedH, H));

  // Verify predict.
  Matrix1 pH1;
  Matrix1 pH2;
  Rot2 Ri = Rot2::fromAngle(1);
  Rot2 predictedRj = measurement.predict(Ri, bias, pH1, pH2);
  // 1 + 0.025 = 1.025
  EXPECT(assert_equal(Rot2::fromAngle(1.025), predictedRj, 1e-9))
  // Ri adds to prediction.
  DOUBLES_EQUAL(1, pH1(0, 0), 1e-9)
  // Bias * dt subtracts from prediction.
  DOUBLES_EQUAL(-0.5, pH2(0, 0), 1e-9)

  // Verify computeError.
  Matrix1 cH1;
  Matrix1 cH2;
  Matrix1 cH3;
  Rot2 Rj = Rot2::fromAngle(2);
  Vector1 e = measurement.computeError(Ri, Rj, bias, cH1, cH2, cH3);
  // estimate - prediction = 2 - 1.025
  DOUBLES_EQUAL(-0.975, e(0), 1e-9)
  // Ri up => error up (less negative)
  DOUBLES_EQUAL(1, cH1(0, 0), 1e-9)
  // Rj up -> error down (more negative)
  DOUBLES_EQUAL(-1, cH2(0, 0), 1e-9)
  // bias up -> error down (more negative), scaled by dt
  DOUBLES_EQUAL(-0.5, cH3(0, 0), 1e-9)

  // Integrate a second IMU measurement.
  measurement.integrate(omega, deltaT);
  DOUBLES_EQUAL(0.1, measurement.deltaR_.theta(), 1e-9)
  expectedH = numericalDerivative11<Rot2, double>(g, bias);
  corrected = measurement.deltaR(bias, H);
  DOUBLES_EQUAL(0.05, corrected.theta(), 1e-9)
  EXPECT(assert_equal(expectedH, H));
}
}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
