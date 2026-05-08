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
  PlanarGyroMeasurement measurement(1.0);
  auto f = [&measurement](const double& bias) {
    return measurement.deltaR(bias, {});
  };

  const double omega = 0.1;
  const double deltaT = 0.5;

  // Check integration.
  measurement.integrate(omega, deltaT);
  // need FRIEND_TEST for this
  EXPECT(assert_equal(0.05, measurement.deltaR_.theta(), 1e-9))
  EXPECT(assert_equal(0.5, measurement.deltaT_, 1e-6))

  const double bias = 0.05;
  Matrix1 H;

  // Check the effect of bias.
  Rot2 corrected = measurement.deltaR(bias, H);
  EXPECT(assert_equal(0.025, corrected.theta(), 1e-9))
  EXPECT(assert_equal(-0.5, H(0, 0), 1e-9))

  // Numeric derivative matches.
  Matrix1 numericH = numericalDerivative11(f, bias);
  EXPECT(assert_equal(-0.5, numericH(0, 0), 1e-9))

  // Integrate a second IMU measurement.
  measurement.integrate(omega, deltaT);
  // need FRIEND_TEST for this
  EXPECT(assert_equal(0.1, measurement.deltaR_.theta(), 1e-9))
  EXPECT(assert_equal(1.0, measurement.deltaT_, 1e-6))

  // Check the effect of bias.
  corrected = measurement.deltaR(bias, H);
  EXPECT(assert_equal(0.05, corrected.theta(), 1e-9))
  EXPECT(assert_equal(-1.0, H(0, 0), 1e-9))

  // Numeric derivative matches.
  numericH = numericalDerivative11(f, bias);
  EXPECT(assert_equal(-1.0, numericH(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, variance) {
  PlanarGyroMeasurement measurement(1.0);
  const double omega = 0.1;
  const double deltaT = 0.5;
  measurement.integrate(omega, deltaT);

  // 1.0 * 0.5 = 0.5
  EXPECT(assert_equal(0.5, measurement.variance()(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, predict) {
  PlanarGyroMeasurement measurement(1.0);
  auto f = [&measurement](const Rot2& r, const double& b) -> Rot2 {
    return measurement.predict(r, b);
  };

  const double omega = 0.1;
  const double deltaT = 0.5;
  measurement.integrate(omega, deltaT);

  Rot2 Ri = Rot2::fromAngle(1);
  const double bias = 0.05;
  Matrix1 H1;
  Matrix1 H2;
  Rot2 predictedRj = measurement.predict(Ri, bias, H1, H2);

  // 1 + 0.025 = 1.025
  EXPECT(assert_equal(1.025, predictedRj.theta(), 1e-9))
  // Ri adds to prediction.
  EXPECT(assert_equal(1.0, H1(0, 0), 1e-9))
  // Bias * dt subtracts from prediction.
  EXPECT(assert_equal(-0.5, H2(0, 0), 1e-9))

  // Numeric derivative matches.
  Matrix1 nH1 = numericalDerivative21(f, Ri, bias);
  Matrix1 nH2 = numericalDerivative22(f, Ri, bias);
  EXPECT(assert_equal(1.0, nH1(0, 0), 1e-9))
  EXPECT(assert_equal(-0.5, nH2(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, computeError) {
  PlanarGyroMeasurement measurement(1.0);
  auto f = [&measurement](const Rot2& r1, const Rot2& r2,
                          const double& b) -> Vector1 {
    return measurement.computeError(r1, r2, b);
  };
  const double omega = 0.1;
  const double deltaT = 0.5;
  measurement.integrate(omega, deltaT);

  Rot2 Ri = Rot2::fromAngle(1);
  Rot2 Rj = Rot2::fromAngle(2);
  const double bias = 0.05;
  Matrix1 H1, H2, H3;
  Vector1 err = measurement.computeError(Ri, Rj, bias, H1, H2, H3);

  // estimate - prediction = 2 - 1.025 = -0.975
  EXPECT(assert_equal(-0.975, err(0), 1e-9))
  // Ri up => error up (less negative)
  EXPECT(assert_equal(1.0, H1(0, 0), 1e-9))
  // Rj up -> error down (more negative)
  EXPECT(assert_equal(-1.0, H2(0, 0), 1e-9))
  // bias up -> error down (more negative), scaled by dt
  EXPECT(assert_equal(-0.5, H3(0, 0), 1e-9))

  // Numeric derivative matches
  Matrix1 nH1 = numericalDerivative31(f, Ri, Rj, bias);
  Matrix1 nH2 = numericalDerivative32(f, Ri, Rj, bias);
  Matrix1 nH3 = numericalDerivative33(f, Ri, Rj, bias);
  EXPECT(assert_equal(1.0, nH1(0, 0), 1e-9))
  EXPECT(assert_equal(-1.0, nH2(0, 0), 1e-9))
  EXPECT(assert_equal(-0.5, nH3(0, 0), 1e-9))
}

}  // namespace gtsam

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
