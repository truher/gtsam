/**
 * Like PreintegratedRotation but for 2d rotations.
 *
 *  @file  PreintegratedPlanarRotation.h
 *  @author joel@truher.org
 **/

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/std_optional_serialization.h>
#include <gtsam/geometry/Pose2.h>

#include "gtsam/dllexport.h"

namespace gtsam {

namespace internal {
/**
 * @brief Function object for incremental rotation.
 * @param measuredOmega The measured angular velocity (as given by the sensor)
 * @param deltaT The time interval over which the rotation is integrated.
 * @param body_P_sensor Optional transform between body and IMU.
 */
struct GTSAM_EXPORT IncrementalPlanarRotation {
  const Vector1& measuredOmega;
  const double deltaT;
  const std::optional<Pose2>& body_P_sensor;

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

/// Parameters for pre-integration:
/// Usage: Create just a single Params and pass a shared pointer to the
/// constructor
struct GTSAM_EXPORT PreintegratedPlanarRotationParams {
  /// Continuous-time "Covariance" of gyroscope measurements
  /// The units for stddev are σ = rad/s/√Hz
  Matrix3 gyroscopeCovariance;
  std::optional<Pose2>
      body_P_sensor;  ///< The pose of the sensor in the body frame

  PreintegratedPlanarRotationParams() : gyroscopeCovariance(I_3x3) {}

  PreintegratedPlanarRotationParams(const Matrix3& gyroscope_covariance,
                                    std::optional<Pose2> body_P_sensor = {})
      : gyroscopeCovariance(gyroscope_covariance),
        body_P_sensor(body_P_sensor) {}

  virtual ~PreintegratedPlanarRotationParams() {}

  virtual void print(const std::string& s) const;
  virtual bool equals(const PreintegratedPlanarRotationParams& other,
                      double tol = 1e-9) const;

  void setGyroscopeCovariance(const Matrix3& cov) { gyroscopeCovariance = cov; }
  void setBodyPSensor(const Pose2& pose) { body_P_sensor = pose; }

  const Matrix3& getGyroscopeCovariance() const { return gyroscopeCovariance; }
  std::optional<Pose2> getBodyPSensor() const { return body_P_sensor; }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(gyroscopeCovariance);
    ar& BOOST_SERIALIZATION_NVP(body_P_sensor);
    }
  }
#endif

#ifdef GTSAM_USE_QUATERNIONS
  // Align if we are using Quaternions
 public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
#endif
};

class GTSAM_EXPORT PreintegratedPlanarRotation {
 public:
  typedef PreintegratedPlanarRotationParams Params;

 protected:
  /// Parameters
  std::shared_ptr<Params> p_;

  double deltaTij_;  ///< Time interval from i to j
  Rot2 deltaRij_;    ///< Preintegrated relative orientation (in frame i)
  Matrix3 delRdelBiasOmega_;  ///< Jacobian of preintegrated rotation w.r.t.
                              ///< angular rate bias

 public:
  /// @name Constructors
  /// @{

  /// Default constructor for serialization
  PreintegratedPlanarRotation() {}

  /// Default constructor, resets integration to zero
  explicit PreintegratedPlanarRotation(const std::shared_ptr<Params>& p)
      : p_(p) {
    resetIntegration();
  }

  /// Explicit initialization of all class members
  PreintegratedPlanarRotation(const std::shared_ptr<Params>& p, double deltaTij,
                              const Rot2& deltaRij,
                              const Matrix3& delRdelBiasOmega)
      : p_(p),
        deltaTij_(deltaTij),
        deltaRij_(deltaRij),
        delRdelBiasOmega_(delRdelBiasOmega) {}

  /// @}

  /// @name Basic utilities
  /// @{

  /// check parameters equality: checks whether shared pointer points to same
  /// Params object.
  bool matchesParamsWith(const PreintegratedPlanarRotation& other) const {
    return p_ == other.p_;
  }
  /// @}

  /// @name Access instance variables
  /// @{
  const std::shared_ptr<Params>& params() const { return p_; }
  const double& deltaTij() const { return deltaTij_; }
  const Rot2& deltaRij() const { return deltaRij_; }
  const Matrix3& delRdelBiasOmega() const { return delRdelBiasOmega_; }
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
  Rot2 biascorrectedDeltaRij(const Vector3& biasOmegaIncr,
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

#ifdef GTSAM_USE_QUATERNIONS
  // Align if we are using Quaternions
 public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
#endif
};

template <>
struct traits<PreintegratedPlanarRotation>
    : public Testable<PreintegratedPlanarRotation> {};

}  // namespace gtsam
