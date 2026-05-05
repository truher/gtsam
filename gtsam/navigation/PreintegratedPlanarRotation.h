/**
 * Like PreintegratedRotation but for 2d rotations.
 *
 *  @file  PreintegratedPlanarRotation.h
 *  @author joel@truher.org
 **/
#pragma once
#include <gtsam/base/Matrix.h>
#include <gtsam/base/std_optional_serialization.h>
#include <gtsam/geometry/Rot2.h>

#include "gtsam/dllexport.h"

namespace gtsam {

class GTSAM_EXPORT PreintegratedPlanarRotation {
 protected:
  double gyroscopeCovariance_;
  double deltaTij_;  ///< Time interval from i to j
  Rot2 deltaRij_;    ///< Preintegrated relative orientation (in frame i)
  Matrix1 delRdelBiasOmega_;  ///< Jacobian of preintegrated rotation w.r.t.
                              ///< angular rate bias

 public:
  /// Default constructor for serialization
  PreintegratedPlanarRotation() {}

  /// Default constructor, resets integration to zero
  explicit PreintegratedPlanarRotation(double gyroscopeCovariance)
      : gyroscopeCovariance_(gyroscopeCovariance) {
    resetIntegration();
  }

  /// Explicit initialization of all class members
  PreintegratedPlanarRotation(double gyroscopeCovariance, double deltaTij,
                              const Rot2& deltaRij,
                              const Matrix1& delRdelBiasOmega)
      : gyroscopeCovariance_(gyroscopeCovariance),
        deltaTij_(deltaTij),
        deltaRij_(deltaRij),
        delRdelBiasOmega_(delRdelBiasOmega) {}

  const double& gyroscopeCovariance() const { return gyroscopeCovariance_; }
  const double& deltaTij() const { return deltaTij_; }
  const Rot2& deltaRij() const { return deltaRij_; }
  const Matrix1& delRdelBiasOmega() const { return delRdelBiasOmega_; }
  void print(const std::string& s) const;
  bool equals(const PreintegratedPlanarRotation& other, double tol) const;

  /// Re-initialize PreintegratedMeasurements
  void resetIntegration();

  /**
   * @brief Calculate an incremental rotation given the gyro measurement and a
   * time interval, and update both deltaTij_ and deltaRij_.
   * @param measuredOmega The measured angular velocity (as given by the sensor)
   * @param bias The biasHat estimate
   * @param deltaT The time interval
   */
  void integrateGyroMeasurement(double measuredOmega, double biasHat,
                                double deltaT);

  /**
   * @brief Return a bias corrected version of the integrated rotation.
   * @param biasOmegaIncr An increment with respect to biasHat used above.
   * @param H optional Jacobian of the correction w.r.t. the bias increment.
   * @note The *key* functionality of this class used in optimizing the bias.
   */
  Rot2 biascorrectedDeltaRij(double biasOmegaIncr,
                             OptionalJacobian<1, 1> H = {}) const;
};
}  // namespace gtsam
