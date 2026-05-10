/**
 * Measurement of a one-dimensional gyro, handles bias.
 * 
 * Useful for high-school robotics competitions,
 * which run robots on the floor, and so measure yaw.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarGyroMeasurement.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once
#include <gtsam/geometry/Rot2.h>

#include "gtsam/dllexport.h"

namespace gtsam {
class GTSAM_EXPORT PlanarGyroMeasurement {
 private:
  /**
   * Published or measured variance of gyroscope measurements.
   * This is white noise in omega, which results in "angle random walk"
   * (ARW) in the integrated rotation measurement.
   * Stddev (σ) unit is rad/s/√Hz.
   * Variance (σ^2) unit is (rad/s)^2/Hz, or rad^2/s.
   */
  const double ARW_;
  /** Incremental rotation */
  const Rot2 deltaR_;
  /** Measurement time interval (s) */
  const double deltaT_;

  PlanarGyroMeasurement(double ARW, Rot2 dr, double dt)
      : ARW_(ARW), deltaR_(dr), deltaT_(dt) {}

 public:
  /**
   * @param ARW "angle random walk" instrument variance (rad^2/s)
   * @param omega average rotation rate during dt (rad/s)
   * @param dt incremental time (s)
   */
  static inline PlanarGyroMeasurement fromRate(double ARW, double omega,
                                               double dt) {
    return PlanarGyroMeasurement(ARW, Rot2::fromAngle(omega * dt), dt);
  }

  /**
   * @param ARW "angle random walk" instrument variance (rad^2/s)
   * @param dr incremental rotation during dt
   * @param dt incremental time (s)
   */
  static inline PlanarGyroMeasurement fromRotation(double ARW, Rot2 dr,
                                                   double dt) {
    return PlanarGyroMeasurement(ARW, dr, dt);
  }

  /**
   * Variance of the measurement (rad^2)
   */
  double variance() const {
    // Integrated white noise => variance scales linearly with time.
    return ARW_ * deltaT_;
  }

  void print(const std::string& s = "Measurements: ") const;
  bool equals(const PlanarGyroMeasurement& expected, double tol = 1e-9) const;

  /**
   * Bias-corrected rotation.
   *
   * @param bias rate (rad/s)
   * @param H derivative of rotation wrt bias.
   */
  Rot2 deltaR(double bias, OptionalJacobian<1, 1> H = {}) const;

  /**
   * Predicted rotation at time j, given rotation and bias at time i.
   *
   * @param Ri rotation at time i (rad)
   * @param bias rate (rad/s)
   * @param H1 dRj/dRi
   * @param H2 dRj/dBias
   */
  Rot2 predict(const Rot2& Ri, double bias, OptionalJacobian<1, 1> H1 = {},
               OptionalJacobian<1, 1> H2 = {}) const;

  /**
   * The error between the predicted and actual rotation (rad)
   *
   * @param Ri rotation at time i (rad)
   * @param Rj rotation at time j (rad)
   * @param bias rate (rad/s)
   * @param H1 dErr/dRi
   * @param H2 dErr/dRj
   * @param H3 dErr/dBias
   */
  double computeError(const Rot2& Ri, const Rot2& Rj, double bias,
                      OptionalJacobian<1, 1> H1 = {},
                      OptionalJacobian<1, 1> H2 = {},
                      OptionalJacobian<1, 1> H3 = {}) const;
};

}  // namespace gtsam