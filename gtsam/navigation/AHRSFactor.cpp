/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  AHRSFactor.cpp
 *  @author Krunal Chande
 *  @author Luca Carlone
 *  @author Frank Dellaert
 *  @date   July 2014
 **/

#include <gtsam/navigation/AHRSFactor.h>
#include <iostream>

using namespace std;

namespace gtsam {

//------------------------------------------------------------------------------
// Inner class PreintegratedMeasurements
//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::print(const string& s) const {
  PreintegratedRotation::print(s);
  cout << "biasHat [" << biasHat_.transpose() << "]" << endl;
  cout << " PreintMeasCov [ " << preintMeasCov_ << " ]" << endl;
}

//------------------------------------------------------------------------------
bool PreintegratedAhrsMeasurements::equals(
    const PreintegratedAhrsMeasurements& other, double tol) const {
  return PreintegratedRotation::equals(other, tol) &&
         equal_with_abs_tol(biasHat_, other.biasHat_, tol);
}

//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::resetIntegration() {
  PreintegratedRotation::resetIntegration();
  preintMeasCov_.setZero();
}

//------------------------------------------------------------------------------
void PreintegratedAhrsMeasurements::integrateMeasurement(
    const Vector3& measuredOmega, double deltaT) {
  Matrix3 Fr;
  PreintegratedRotation::integrateGyroMeasurement(measuredOmega, biasHat_,
                                                  deltaT, &Fr);

  // First order uncertainty propagation
  // The deltaT allows to pass from continuous time noise to discrete time
  // noise. Comparing with the IMUFactor.cpp implementation, the latter is an
  // approximation for C * (wCov / dt) * C.transpose(), with C \approx I * dt.
  preintMeasCov_ = Fr * preintMeasCov_ * Fr.transpose() + p().gyroscopeCovariance * deltaT;
}

//------------------------------------------------------------------------------
Vector3 PreintegratedAhrsMeasurements::predict(const Vector3& bias,
    OptionalJacobian<3,3> H) const {
  const Vector3 biasOmegaIncr = bias - biasHat_;
  const Rot3 biascorrected = biascorrectedDeltaRij(biasOmegaIncr, H);
  Matrix3 D_omega_biascorrected;
  const Vector3 omega = Rot3::Logmap(biascorrected, H ? &D_omega_biascorrected : 0);
  if (H) (*H) = D_omega_biascorrected * (*H);
  return omega;
}
//------------------------------------------------------------------------------
Vector PreintegratedAhrsMeasurements::DeltaAngles(
    const Vector& msr_gyro_t, const double msr_dt,
    const Vector3& delta_angles) {

  // Note: all delta terms refer to an IMU\sensor system at t0

  // Calculate the corrected measurements using the Bias object
  Vector body_t_omega_body = msr_gyro_t;

  Rot3 R_t_to_t0 = Rot3::Expmap(delta_angles);

  R_t_to_t0 = R_t_to_t0 * Rot3::Expmap(body_t_omega_body * msr_dt);
  return Rot3::Logmap(R_t_to_t0);
}

//------------------------------------------------------------------------------
// AHRSFactor methods
//------------------------------------------------------------------------------
AHRSFactor::AHRSFactor(
    Key rot_i, Key rot_j, Key bias,
    const PreintegratedAhrsMeasurements& preintegratedMeasurements)
    : Base(noiseModel::Gaussian::Covariance(
               preintegratedMeasurements.preintMeasCov_),
           rot_i, rot_j, bias),
      _PIM_(preintegratedMeasurements) {}

gtsam::NonlinearFactor::shared_ptr AHRSFactor::clone() const {
//------------------------------------------------------------------------------
  return std::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
}

//------------------------------------------------------------------------------
void AHRSFactor::print(const string& s,
    const KeyFormatter& keyFormatter) const {
  cout << s << "AHRSFactor(" << keyFormatter(this->key<1>()) << ","
      << keyFormatter(this->key<2>()) << "," << keyFormatter(this->key<3>()) << ",";
  _PIM_.print("  preintegrated measurements:");
  noiseModel_->print("  noise model: ");
}

//------------------------------------------------------------------------------
bool AHRSFactor::equals(const NonlinearFactor& other, double tol) const {
  const This *e = dynamic_cast<const This*>(&other);
  return e != nullptr && Base::equals(*e, tol) && _PIM_.equals(e->_PIM_, tol);
}

//------------------------------------------------------------------------------
Vector AHRSFactor::evaluateError(const Rot3& Ri, const Rot3& Rj,
    const Vector3& bias, OptionalMatrixType H1,
    OptionalMatrixType H2, OptionalMatrixType H3) const {

  // Do bias correction, if (H3) will contain 3*3 derivative used below
  const Vector3 biascorrectedOmega = _PIM_.predict(bias, H3);

  // Coriolis term
  const Vector3 coriolis = _PIM_.integrateCoriolis(Ri);
  const Vector3 correctedOmega = biascorrectedOmega - coriolis;

  // Prediction
  const Rot3 correctedDeltaRij = Rot3::Expmap(correctedOmega);

  // Get error between actual and prediction
  const Rot3 actualRij = Ri.between(Rj);
  const Rot3 fRrot = correctedDeltaRij.between(actualRij);
  Vector3 fR = Rot3::Logmap(fRrot);

  // Terms common to derivatives
  const Matrix3 D_cDeltaRij_cOmega = Rot3::ExpmapDerivative(correctedOmega);
  const Matrix3 D_fR_fRrot = Rot3::LogmapDerivative(fR);

  if (H1) {
    // dfR/dRi
    H1->resize(3, 3);
    Matrix3 D_coriolis = -D_cDeltaRij_cOmega * skewSymmetric(coriolis);
    (*H1)
        << D_fR_fRrot * (-actualRij.transpose() - fRrot.transpose() * D_coriolis);
  }

  if (H2) {
    // dfR/dPosej
    H2->resize(3, 3);
    (*H2) << D_fR_fRrot;
  }

  if (H3) {
    // dfR/dBias, note H3 contains derivative of predict
    const Matrix3 JbiasOmega = D_cDeltaRij_cOmega * (*H3);
    H3->resize(3, 3);
    (*H3) << D_fR_fRrot * (-fRrot.transpose() * JbiasOmega);
  }

  Vector error(3);
  error << fR;
  return error;
}

//------------------------------------------------------------------------------
Rot3 AHRSFactor::Predict(const Rot3& rot_i, const Vector3& bias,
                         const PreintegratedAhrsMeasurements& pim) {
  const Vector3 biascorrectedOmega = pim.predict(bias);

  // Coriolis term
  const Vector3 coriolis = pim.integrateCoriolis(rot_i);

  const Vector3 correctedOmega = biascorrectedOmega - coriolis;
  const Rot3 correctedDeltaRij = Rot3::Expmap(correctedOmega);

  return rot_i.compose(correctedDeltaRij);
}

//------------------------------------------------------------------------------
AHRSFactor::AHRSFactor(Key rot_i, Key rot_j, Key bias,
                       const PreintegratedAhrsMeasurements& pim,
                       const Vector3& omegaCoriolis,
                       const std::optional<Pose3>& body_P_sensor)
    : Base(noiseModel::Gaussian::Covariance(pim.preintMeasCov_), rot_i, rot_j,
           bias),
      _PIM_(pim) {
  auto p = std::make_shared<PreintegratedAhrsMeasurements::Params>(pim.p());
  p->body_P_sensor = body_P_sensor;
  _PIM_.p_ = p;
}

//------------------------------------------------------------------------------
Rot3 AHRSFactor::predict(const Rot3& rot_i, const Vector3& bias,
                         const PreintegratedAhrsMeasurements& pim,
                         const Vector3& omegaCoriolis,
                         const std::optional<Pose3>& body_P_sensor) {
  auto p = std::make_shared<PreintegratedAhrsMeasurements::Params>(pim.p());
  p->omegaCoriolis = omegaCoriolis;
  p->body_P_sensor = body_P_sensor;
  PreintegratedAhrsMeasurements newPim = pim;
  newPim.p_ = p;
  return Predict(rot_i, bias, newPim);
}

}  // namespace gtsam
