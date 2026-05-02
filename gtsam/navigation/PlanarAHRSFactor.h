/**
 * Like AHRSFactor, except:
 * 
 * * using Rot2 instead of Rot3
 * * without the preintegrator: measurements are preintegrated by the gyro hardware
 * * without coriolis correction: rotating reference frame is irrelevant
 * * without deprecated v4 stuff
 * * without the body transform: translation doesn't matter, rotation is always identity.
 *
 * This factor is useful for high-school robotics competitions,
 * which run robots on the floor: they really only care about yaw.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarAHRSFactor.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once

/* GTSAM includes */
#include <gtsam/geometry/Pose2.h>
#include <gtsam/navigation/PreintegratedPlanarRotation.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>

namespace gtsam {

class GTSAM_EXPORT PreintegratedPlanarAhrsMeasurements
    : public PreintegratedPlanarRotation {
 protected:
  Vector1 biasHat_;  ///< Angular rate bias values used during preintegration.
  Matrix1 preintMeasCov_;  ///< Covariance matrix of the preintegrated
                           ///< measurements (first-order propagation from
                           ///< *measurementCovariance*)

  friend class PlanarAHRSFactor;

 public:
  /// Default constructor, only for serialization and wrappers
  PreintegratedPlanarAhrsMeasurements() {}

  /**
   *  Default constructor, initialize with no measurements
   *  @param bias Current estimate of rotation rate biases
   */
  PreintegratedPlanarAhrsMeasurements(const std::shared_ptr<Params>& p,
                                      const Vector1& biasHat)
      : PreintegratedPlanarRotation(p), biasHat_(biasHat) {
    resetIntegration();
  }

  /**
   *  Non-Default constructor, initialize with measurements
   *  @param p: Parameters for AHRS pre-integration
   *  @param bias_hat: Current estimate of rotation rate biases
   *  @param deltaTij: Delta time in pre-integration
   *  @param deltaRij: Delta rotation in pre-integration
   *  @param delRdelBiasOmega: Jacobian of rotation wrt gyro bias
   *  @param preint_meas_cov: Pre-integration covariance
   */
  PreintegratedPlanarAhrsMeasurements(const std::shared_ptr<Params>& p,
                                      const Vector1& bias_hat,
                                      double deltaTij,
                                      const Rot2& deltaRij,
                                      const Matrix1& delRdelBiasOmega,
                                      const Matrix1& preint_meas_cov)
      : PreintegratedPlanarRotation(p, deltaTij, deltaRij, delRdelBiasOmega),
        biasHat_(bias_hat),
        preintMeasCov_(preint_meas_cov) {}

  Params& p() const { return *std::static_pointer_cast<Params>(p_); }
  const Vector1& biasHat() const { return biasHat_; }
  const Matrix1& preintMeasCov() const { return preintMeasCov_; }

  /// print
  void print(const std::string& s = "Preintegrated Measurements: ") const;

  /// equals
  bool equals(const PreintegratedPlanarAhrsMeasurements& expected,
              double tol = 1e-9) const;

  /// Reset integrated quantities to zero
  void resetIntegration();

  /**
   * Add a single gyroscope measurement to the preintegration.
   *
   * @param measuredOmega Measured angular velocity (as given by the sensor)
   * @param deltaT Time step
   */
  void integrateMeasurement(const Vector1& measuredOmega, double deltaT);

  /**
   * Predict the orientation at time j, given orientation and bias at time i.
   * @param Ri orientation at time i
   * @param bias gyroscope bias
   * @param H1 optional Jacobian wrt Ri
   * @param H2 optional Jacobian wrt bias
   * @return predicted orientation at time j
   */
  Rot2 predict(const Rot2& Ri,
               const Vector1& bias,
               gtsam::OptionalJacobian<3, 3> H1 = {},
               gtsam::OptionalJacobian<3, 3> H2 = {}) const;

  /**
   * Calculate the error between the predicted and actual rotation.
   * @param Ri The orientation at time i
   * @param Rj The orientation at time j
   * @param bias The gyroscope bias
   * @param H1 Optional Jacobian of the error with respect to Ri
   * @param H2 Optional Jacobian of the error with respect to Rj
   * @param H3 Optional Jacobian of the error with respect to bias
   * @return A 3D vector containing the rotation error.
   */
  Vector1 computeError(const Rot2& Ri,
                       const Rot2& Rj,
                       const Vector1& bias,
                       gtsam::OptionalJacobian<3, 3> H1 = {},
                       gtsam::OptionalJacobian<3, 3> H2 = {},
                       gtsam::OptionalJacobian<3, 3> H3 = {}) const;

  /// @deprecated constructor, but used in tests.
  PreintegratedPlanarAhrsMeasurements(const Vector1& biasHat,
                                      const Matrix1& measuredOmegaCovariance)
      : PreintegratedPlanarRotation(std::make_shared<Params>()), biasHat_(biasHat) {
    p_->gyroscopeCovariance = measuredOmegaCovariance;
    resetIntegration();
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegratedPlanarRotation);
    ar& BOOST_SERIALIZATION_NVP(p_);
    ar& BOOST_SERIALIZATION_NVP(biasHat_);
  }
#endif
};

/**
 * See AHRSFactor.
 */
class GTSAM_EXPORT PlanarAHRSFactor : public NoiseModelFactorN<Rot2, Rot2, Vector1> {
  typedef PlanarAHRSFactor This;
  typedef NoiseModelFactorN<Rot2, Rot2, Vector1> Base;

  PreintegratedPlanarAhrsMeasurements _PIM_;

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /** Shorthand for a smart pointer to a factor */
#if !defined(_MSC_VER) && __GNUC__ == 4 && __GNUC_MINOR__ > 5
  typedef typename std::shared_ptr<PlanarAHRSFactor> shared_ptr;
#else
  typedef std::shared_ptr<PlanarAHRSFactor> shared_ptr;
#endif

  /** Default constructor - only use for serialization */
  PlanarAHRSFactor() {}

  /**
   * Constructor
   * @param rot_i previous rot key
   * @param rot_j current rot key
   * @param bias  previous bias key
   * @param pim preintegrated measurements
   */
  PlanarAHRSFactor(Key rot_i, Key rot_j, Key bias,
             const PreintegratedPlanarAhrsMeasurements& pim);

  ~PlanarAHRSFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override;

  /// print
  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override;

  /// equals
  bool equals(const NonlinearFactor&, double tol = 1e-9) const override;

  /// Access the preintegrated measurements.
  const PreintegratedPlanarAhrsMeasurements& preintegratedMeasurements() const {
    return _PIM_;
  }

  /** implement functions needed to derive from Factor */

  /// vector of errors
  Vector evaluateError(const Rot2& Ri,
                       const Rot2& Rj,
                       const Vector1& bias,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2,
                       OptionalMatrixType H3) const override;

  /// @deprecated constructor, but used in tests.
  PlanarAHRSFactor(Key rot_i, Key rot_j, Key bias,
             const PreintegratedPlanarAhrsMeasurements& pim);

  /// @deprecated static function, but used in tests.
  static Rot2 predict(const Rot2& Ri,
                      const Vector1& bias,
                      const PreintegratedPlanarAhrsMeasurements& pim);

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    // NoiseModelFactor3 instead of NoiseModelFactorN for backward compatibility
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor3", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(_PIM_);
  }
#endif
};
// PlanarAHRSFactor

}  // namespace gtsam