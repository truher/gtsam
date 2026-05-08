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
  // Published or measured continuous-time variance of gyroscope
  // measurements.
  // This is white noise in omega, which results in "angle random walk"
  // (ARW) in the integrated measurement.
  // The units for stddev are σ = rad/s/√Hz.
  // Note variance should be σ^2 so (rad/s)^2/Hz or rad^2/s
  const double ARW_;
  // Integrated time interval (sec)
  double deltaT_;
  // Integrated rotation
  Rot2 deltaR_;

  friend class PlanarGyroFactor;
  FRIEND_TEST(PlanarGyroFactor, PlanarGyroMeasurement)
  FRIEND_TEST(PlanarGyroFactor, FirstOrderPlanarGyro)
  FRIEND_TEST(PlanarGyroMeasurement, integrateGyroMeasurement)

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
   * Bias corrected integrated rotation.
   *
   * @param bias rate rad/s
   * @param H optional Jacobian of the correction w.r.t. the bias.
   */
  Rot2 biascorrectedDeltaR(double bias, OptionalJacobian<1, 1> H = {}) const;

  /**
   * Adds a single gyroscope measurement to the preintegration.
   *
   * Calculates an incremental rotation given the gyro measurement and a
   * time interval.  Updates both deltaTij_ and deltaRij_.
   *
   * @param omega rotation rate (rad/s)
   * @param dt time step (s)
   */
  void integrateMeasurement(double omega, double dt);

  /**
   * Predict the orientation at time j, given orientation and bias at time i.
   *
   * @param Ri rotation at time i (rad)
   * @param bias rate (rad/s)
   * @param H1 optional Jacobian wrt Ri
   * @param H2 optional Jacobian wrt bias
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
   * @param H1 Optional Jacobian of the error with respect to Ri
   * @param H2 Optional Jacobian of the error with respect to Rj
   * @param H3 Optional Jacobian of the error with respect to bias
   * @return The rotation error.
   */
  Vector1 computeError(const Rot2& Ri, const Rot2& Rj, double bias,
                       gtsam::OptionalJacobian<1, 1> H1 = {},
                       gtsam::OptionalJacobian<1, 1> H2 = {},
                       gtsam::OptionalJacobian<1, 1> H3 = {}) const;
};

}  // namespace gtsam