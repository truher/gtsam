/**
 * Like AHRSFactor, except:
 *
 * * using Rot2 instead of Rot3
 * * without the preintegrator: measurements are preintegrated by the gyro
 * hardware
 * * without coriolis correction: rotating reference frame is irrelevant
 * * without deprecated v4 stuff
 * * without the body transform: translation doesn't matter, rotation is always
 * identity.
 *
 * This factor is useful for high-school robotics competitions,
 * which run robots on the floor: they really only care about yaw.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarGyroMeasurement.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/std_optional_serialization.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>

#include "gtsam/dllexport.h"

namespace gtsam {

class GTSAM_EXPORT PlanarGyroMeasurement {
 protected:
  // Published or measured continuous-time "Covariance" of gyroscope
  // measurements.
  // This is white noise in omega, which results in "angle random walk"
  // in the integrated measurement.
  // The units for stddev are σ = rad/s/√Hz.
  // Note variance should be σ^2 so (rad/s)^2/Hz or rad^2/s
  const double gyroscopeCovariance_;

  // Time interval from i to j
  double deltaTij_;
  // Rotation of j relative to i
  Rot2 deltaRij_;

  friend class PlanarGyroFactor;

 public:
  PlanarGyroMeasurement(double gyroscopeCovariance)
      : gyroscopeCovariance_(gyroscopeCovariance),
        deltaTij_(0.0),
        deltaRij_(Rot2()) {}

  const double& gyroscopeCovariance() const { return gyroscopeCovariance_; }
  const double& deltaTij() const { return deltaTij_; }
  const Rot2& deltaRij() const { return deltaRij_; }
  const Matrix1 delRdelBiasOmega() const {
    Matrix1 m;
    m << -deltaTij_;
    return m;
  }
  const Matrix1 preintMeasCov() const {
    Matrix1 m;
    m << gyroscopeCovariance_ * deltaTij_;
    return m;
  }

  void print(const std::string& s = "Preintegrated Measurements: ") const;
  bool equals(const PlanarGyroMeasurement& expected, double tol = 1e-9) const;

  /**
   * @brief Return a bias corrected version of the integrated rotation.
   * @param biasOmegaIncr An increment with respect to biasHat used above.
   * @param H optional Jacobian of the correction w.r.t. the bias increment.
   * @note The *key* functionality of this class used in optimizing the bias.
   */
  Rot2 biascorrectedDeltaRij(double biasOmegaIncr,
                             OptionalJacobian<1, 1> H = {}) const;

  /**
   * Adds a single gyroscope measurement to the preintegration.
   *
   * Calculates an incremental rotation given the gyro measurement and a
   * time interval, and update both deltaTij_ and deltaRij_.
   *
   * @param measuredOmega Measured angular velocity (as given by the sensor)
   * @param deltaT Time step
   */
  void integrateMeasurement(double measuredOmega, double deltaT);

  /**
   * Predict the orientation at time j, given orientation and bias at time i.
   * @param Ri orientation at time i
   * @param bias gyroscope bias
   * @param H1 optional Jacobian wrt Ri
   * @param H2 optional Jacobian wrt bias
   * @return predicted orientation at time j
   */
  Rot2 predict(const Rot2& Ri, double bias,
               gtsam::OptionalJacobian<1, 1> H1 = {},
               gtsam::OptionalJacobian<1, 1> H2 = {}) const;

  /**
   * Calculate the error between the predicted and actual rotation.
   * @param Ri The orientation at time i
   * @param Rj The orientation at time j
   * @param bias The gyroscope bias
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