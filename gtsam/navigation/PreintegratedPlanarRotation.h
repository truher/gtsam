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

namespace internal {
/**
 * @brief Function object for incremental rotation.
 * @param measuredOmega The measured angular velocity (as given by the sensor)
 * @param deltaT The time interval over which the rotation is integrated.
 */
struct GTSAM_EXPORT IncrementalPlanarRotation {
  const Vector1& measuredOmega;
  const double deltaT;

  /**
   * @brief Integrate angular velocity, but corrected by bias.
   * @param bias The bias estimate
   * @param H_bias Jacobian of the rotation w.r.t. bias.
   * @return The incremental rotation
   */
  Rot2 operator()(const Vector1& bias,
                  OptionalJacobian<1, 1> H_bias = {}) const;
};

}  // namespace internal

class GTSAM_EXPORT PreintegratedPlanarRotation {
 protected:
  /// Parameters
  double gyroscopeCovariance_;
  double deltaTij_;  ///< Time interval from i to j
  Rot2 deltaRij_;    ///< Preintegrated relative orientation (in frame i)
  Matrix1 delRdelBiasOmega_;  ///< Jacobian of preintegrated rotation w.r.t.
                              ///< angular rate bias

 public:
  /// @name Constructors
  /// @{

  /// Default constructor for serialization
  PreintegratedPlanarRotation() {}

  /// Default constructor, resets integration to zero
  explicit PreintegratedPlanarRotation(double gyroscopeCovariance)
      : gyroscopeCovariance_(gyroscopeCovariance) {
    resetIntegration();
  }

  /// Explicit initialization of all class members
  PreintegratedPlanarRotation(double gyroscopeCovariance,
                              double deltaTij,
                              const Rot2& deltaRij,
                              const Matrix1& delRdelBiasOmega)
      : gyroscopeCovariance_(gyroscopeCovariance),
        deltaTij_(deltaTij),
        deltaRij_(deltaRij),
        delRdelBiasOmega_(delRdelBiasOmega) {}

  /// @}

  /// @name Basic utilities
  /// @{

  /// check parameters equality: checks whether shared pointer points to same
  /// Params object.
  bool matchesParamsWith(const PreintegratedPlanarRotation& other) const {
    return abs(gyroscopeCovariance_ - other.gyroscopeCovariance_ < 1e-9);
  }
  /// @}

  /// @name Access instance variables
  /// @{
  const double& gyroscopeCovariance() const { return gyroscopeCovariance_; }
  const double& deltaTij() const { return deltaTij_; }
  const Rot2& deltaRij() const { return deltaRij_; }
  const Matrix1& delRdelBiasOmega() const { return delRdelBiasOmega_; }
  /// @}

  /// @name Testable
  /// @{
  void print(const std::string& s) const;
  bool equals(const PreintegratedPlanarRotation& other, double tol) const;
  /// @}

  /// @name Main functionality
  /// @{

  /// Re-initialize PreintegratedMeasurements
  void resetIntegration();

  /**
   * @brief Calculate an incremental rotation given the gyro measurement and a
   * time interval, and update both deltaTij_ and deltaRij_.
   * @param measuredOmega The measured angular velocity (as given by the sensor)
   * @param bias The biasHat estimate
   * @param deltaT The time interval
   * @param F optional Jacobian of internal compose, used in AhrsFactor.
   */
  void integrateGyroMeasurement(const Vector1& measuredOmega,
                                const Vector1& biasHat, double deltaT,
                                OptionalJacobian<1, 1> F = {});

  /**
   * @brief Return a bias corrected version of the integrated rotation.
   * @param biasOmegaIncr An increment with respect to biasHat used above.
   * @param H optional Jacobian of the correction w.r.t. the bias increment.
   * @note The *key* functionality of this class used in optimizing the bias.
   */
  Rot2 biascorrectedDeltaRij(const Vector1& biasOmegaIncr,
                             OptionalJacobian<1, 1> H = {}) const;


 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {  // NOLINT
    ar& BOOST_SERIALIZATION_NVP(p_);
    ar& BOOST_SERIALIZATION_NVP(deltaTij_);
    ar& BOOST_SERIALIZATION_NVP(deltaRij_);
    ar& BOOST_SERIALIZATION_NVP(delRdelBiasOmega_);
  }
#endif

};

template <>
struct traits<PreintegratedPlanarRotation>
    : public Testable<PreintegratedPlanarRotation> {};

}  // namespace gtsam
