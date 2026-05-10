/**
 * @file   testPlanarGyroMeasurement.cpp
 * @brief  Unit tests for PlanarGyroMeasurement
 * @author joel@truher.org
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/PlanarGyroMeasurement.h>

namespace gtsam {

TEST(PlanarGyroMeasurement, fromRate) {
  const double arw = 1.0;
  const double omega = 0.1;
  const double dt = 0.5;
  PlanarGyroMeasurement x = PlanarGyroMeasurement::fromRate(arw, omega, dt);

  // Check the effect of bias.
  const double bias = 0.05;
  Matrix1 H;
  Rot2 corrected = x.deltaR(bias, H);
  EXPECT(assert_equal(0.025, corrected.theta(), 1e-9))
  EXPECT(assert_equal(-0.5, H(0, 0), 1e-9))

  // Numeric derivative matches.
  auto f = [&x](const double& bias) { return x.deltaR(bias, {}); };
  Matrix1 numericH = numericalDerivative11(f, bias);
  EXPECT(assert_equal(-0.5, numericH(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, fromRotation) {
  const double arw = 1.0;
  const Rot2 dr = 0.05;
  const double dt = 0.5;
  PlanarGyroMeasurement x = PlanarGyroMeasurement::fromRotation(arw, dr, dt);
  const double bias = 0.05;
  Matrix1 H;
  Rot2 corrected = x.deltaR(bias, H);
  EXPECT(assert_equal(0.025, corrected.theta(), 1e-9))
  EXPECT(assert_equal(-0.5, H(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, variance) {
  const double arw = 1.0;
  const double omega = 0.1;
  const double dt = 0.5;
  PlanarGyroMeasurement x = PlanarGyroMeasurement::fromRate(arw, omega, dt);

  // 1.0 * 0.5 = 0.5
  EXPECT(assert_equal(0.5, x.variance(), 1e-9))
}

TEST(PlanarGyroMeasurement, predict) {
  const double arw = 1.0;
  const double omega = 0.1;
  const double dt = 0.5;
  PlanarGyroMeasurement x = PlanarGyroMeasurement::fromRate(arw, omega, dt);

  // Check prediction.
  Rot2 Ri = Rot2::fromAngle(1);
  const double bias = 0.05;
  Matrix1 H1, H2;
  Rot2 predictedRj = x.predict(Ri, bias, H1, H2);

  // 1 + 0.025 = 1.025
  EXPECT(assert_equal(1.025, predictedRj.theta(), 1e-9))
  // Ri adds to prediction.
  EXPECT(assert_equal(1.0, H1(0, 0), 1e-9))
  // Bias * dt subtracts from prediction.
  EXPECT(assert_equal(-0.5, H2(0, 0), 1e-9))

  // Numeric derivative matches.
  auto f = [&x](const Rot2& r, const double& b) -> Rot2 {
    return x.predict(r, b);
  };
  Matrix1 nH1 = numericalDerivative21(f, Ri, bias);
  Matrix1 nH2 = numericalDerivative22(f, Ri, bias);
  EXPECT(assert_equal(1.0, nH1(0, 0), 1e-9))
  EXPECT(assert_equal(-0.5, nH2(0, 0), 1e-9))
}

TEST(PlanarGyroMeasurement, computeError) {
  const double arw = 1.0;
  const double omega = 0.1;
  const double dt = 0.5;
  PlanarGyroMeasurement x = PlanarGyroMeasurement::fromRate(arw, omega, dt);

  // Check error.
  Rot2 Ri = Rot2::fromAngle(1);
  Rot2 Rj = Rot2::fromAngle(2);
  const double bias = 0.05;
  Matrix1 H1, H2, H3;
  double err = x.computeError(Ri, Rj, bias, H1, H2, H3);

  // estimate - prediction = 2 - 1.025 = -0.975
  EXPECT(assert_equal(-0.975, err, 1e-9))
  // Ri up => error up (less negative)
  EXPECT(assert_equal(1.0, H1(0, 0), 1e-9))
  // Rj up -> error down (more negative)
  EXPECT(assert_equal(-1.0, H2(0, 0), 1e-9))
  // bias up -> error down (more negative), scaled by dt
  EXPECT(assert_equal(-0.5, H3(0, 0), 1e-9))

  // Numeric derivative matches
  auto f = [&x](const Rot2& r1, const Rot2& r2, const double& b) -> double {
    return x.computeError(r1, r2, b);
  };
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
