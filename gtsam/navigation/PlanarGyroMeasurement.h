/**
 * Measurement of a one-dimensional gyro.
 * 
 * Integrates omega over time, corrects for bias.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarGyroMeasurement.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Rot2.h>

#include <optional>

#include "gtsam/dllexport.h"

namespace gtsam {

class GTSAM_EXPORT PlanarGyroMeasurement {
 private:
  // Published or measured variance of gyroscope measurements.
  // This is white noise in omega, which results in "angle random walk"
  // (ARW) in the integrated measurement.
  // Stddev (σ) unit is rad/s/√Hz.
  // Variance σ^2 unit is (rad/s)^2/Hz, or rad^2/s.
  const double ARW_;
  // Integrated time interval (sec).
  double deltaT_;
  // Integrated rotation.
  Rot2 deltaR_;

  friend class PlanarGyroFactor;
  FRIEND_TEST(PlanarGyroFactor, PlanarGyroMeasurement)
  FRIEND_TEST(PlanarGyroFactor, FirstOrderPlanarGyro)
  FRIEND_TEST(PlanarGyroMeasurement, integrate)

 public:
  PlanarGyroMeasurement(double ARW)
      : ARW_(ARW), deltaT_(0.0), deltaR_(Rot2()) {}

  // Variance of the integrated measurement (rad^2)
  const Matrix1 variance() const {
    // Integrating white noise => variance scales linearly with time.
    Matrix1 m;
    m << ARW_ * deltaT_;
    return m;
  }

  void print(const std::string& s = "Preintegrated Measurements: ") const;
  bool equals(const PlanarGyroMeasurement& expected, double tol = 1e-9) const;

  /**
   * Bias-corrected integrated rotation.
   *
   * @param bias rate (rad/s)
   * @param H derivative of rotation wrt bias.
   */
  Rot2 deltaR(double bias, OptionalJacobian<1, 1> H = {}) const;

  /**
   * Calculates an incremental rotation given the measurement
   * and time interval.  Updates both deltaT_ and deltaR_.
   *
   * @param omega rotation rate (rad/s)
   * @param dt time step (s)
   */
  void integrate(double omega, double dt);

  /**
   * Predict the orientation at time j, given orientation and bias at time i.
   *
   * @param Ri rotation at time i (rad)
   * @param bias rate (rad/s)
   * @param H1 derivative of prediction wrt Ri
   * @param H2 derivative of prediction wrt bias
   * @return predicted orientation at time j
   */
  Rot2 predict(const Rot2& Ri, double bias,
               gtsam::OptionalJacobian<1, 1> H1 = {},
               gtsam::OptionalJacobian<1, 1> H2 = {}) const;

  /**
   * Calculate the error between the predicted and actual rotation.
   *
   * @param Ri rotation at time i (rad)
   * @param Rj rotation at time j (rad)
   * @param bias rate (rad/s)
   * @param H1 derivative of error wrt Ri
   * @param H2 derivative of error wrt Rj
   * @param H3 derivative of error wrt bias
   * @return Rotation error (rad)
   */
  Vector1 computeError(const Rot2& Ri, const Rot2& Rj, double bias,
                       gtsam::OptionalJacobian<1, 1> H1 = {},
                       gtsam::OptionalJacobian<1, 1> H2 = {},
                       gtsam::OptionalJacobian<1, 1> H3 = {}) const;
};

}  // namespace gtsam