/**
 * @file PlanarAHRSFactor.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */

#include <gtsam/navigation/PlanarAHRSFactor.h>

#include <iostream>

using namespace std;

namespace gtsam {

//------------------------------------------------------------------------------
// Inner class PreintegratedMeasurements
//------------------------------------------------------------------------------
void PreintegratedPlanarAhrsMeasurements::print(const string& s) const {
  PreintegratedPlanarRotation::print(s);
  cout << "biasHat [" << biasHat_.transpose() << "]" << endl;
  cout << " PreintMeasCov [ " << preintMeasCov_ << " ]" << endl;
}

//------------------------------------------------------------------------------
bool PreintegratedPlanarAhrsMeasurements::equals(
    const PreintegratedPlanarAhrsMeasurements& other, double tol) const {
  return PreintegratedPlanarRotation::equals(other, tol) &&
         equal_with_abs_tol(biasHat_, other.biasHat_, tol);
}

//------------------------------------------------------------------------------
void PreintegratedPlanarAhrsMeasurements::resetIntegration() {
  PreintegratedPlanarRotation::resetIntegration();
  preintMeasCov_.setZero();
}

//------------------------------------------------------------------------------
void PreintegratedPlanarAhrsMeasurements::integrateMeasurement(
    const Vector1& measuredOmega, double deltaT) {
  // 1. integrate
  // Fr is the Jacobian of the new preintegrated rotation w.r.t. the previous
  // one.
  Matrix1 Fr;
  PreintegratedPlanarRotation::integrateGyroMeasurement(measuredOmega, biasHat_,
                                                        deltaT, &Fr);

  // 2. Calculate noise in the body frame
  Matrix1 SigmaBody = p().gyroscopeCovariance;

  // First order uncertainty propagation:
  //   new_cov = Fr * old_cov * Fr.transpose() + new_noise
  // The deltaT allows to pass from continuous time noise to discrete time
  // noise. Comparing with the IMUFactor.cpp implementation, the latter is an
  // approximation for C * (wCov / dt) * C.transpose(), with C \approx I * dt.
  preintMeasCov_ = Fr * preintMeasCov_ * Fr.transpose() + SigmaBody * deltaT;
}

//------------------------------------------------------------------------------
Rot2 PreintegratedPlanarAhrsMeasurements::predict(
    const Rot2& Ri,
    const Vector1& bias,
    gtsam::OptionalJacobian<1, 1> H1,
    gtsam::OptionalJacobian<1, 1> H2) const {
  // Use H2 as an in/out parameter to hold the Jacobian of the bias-corrected
  // rotation w.r.t. the bias increment. This is an efficient C++ pattern.
  const Vector1 biasOmegaIncr = bias - biasHat_;
  const Rot2 biascorrected = this->biascorrectedDeltaRij(biasOmegaIncr, H2);

  return Ri.compose(biascorrected, H1);
}

//------------------------------------------------------------------------------
Vector1 PreintegratedPlanarAhrsMeasurements::computeError(
    const Rot2& Ri,
    const Rot2& Rj,
    const Vector1& bias,
    gtsam::OptionalJacobian<1, 1> H1,
    gtsam::OptionalJacobian<1, 1> H2,
    gtsam::OptionalJacobian<1, 1> H3) const {
  // Predict orientation at time j
  Matrix1 D_predict_Ri, D_predict_bias;
  Rot2 predicted_Rj = predict(Ri, bias, H1 ? &D_predict_Ri : nullptr,
                              H3 ? &D_predict_bias : nullptr);

  // Compute the error vector: log(Rj.inverse() * predicted_Rj)
  Matrix1 D_error_Rj, D_error_predict;
  Vector1 error = Rj.logmap(predicted_Rj, H2 ? &D_error_Rj : nullptr,
                            H1 || H3 ? &D_error_predict : nullptr);

  // Jacobians using the chain rule
  if (H1) *H1 = D_error_predict * D_predict_Ri;
  if (H2) *H2 = D_error_Rj;
  if (H3) *H3 = D_error_predict * D_predict_bias;

  return error;
}

//------------------------------------------------------------------------------
// AHRSFactor methods
//------------------------------------------------------------------------------
PlanarAHRSFactor::PlanarAHRSFactor(
    Key rot_i, Key rot_j, Key bias,
    const PreintegratedPlanarAhrsMeasurements& pim)
    : Base(noiseModel::Gaussian::Covariance(pim.preintMeasCov_), rot_i, rot_j,
           bias),
      _PIM_(pim) {}

gtsam::NonlinearFactor::shared_ptr PlanarAHRSFactor::clone() const {
  //------------------------------------------------------------------------------
  return std::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
}

//------------------------------------------------------------------------------
void PlanarAHRSFactor::print(const string& s,
                             const KeyFormatter& keyFormatter) const {
  cout << s << "PlanarAHRSFactor(" << keyFormatter(this->key<1>()) << ","
       << keyFormatter(this->key<2>()) << "," << keyFormatter(this->key<3>())
       << ",";
  _PIM_.print("  preintegrated measurements:");
  noiseModel_->print("  noise model: ");
}

//------------------------------------------------------------------------------
bool PlanarAHRSFactor::equals(const NonlinearFactor& other, double tol) const {
  const This* e = dynamic_cast<const This*>(&other);
  return e != nullptr && Base::equals(*e, tol) && _PIM_.equals(e->_PIM_, tol);
}

//------------------------------------------------------------------------------
Vector PlanarAHRSFactor::evaluateError(const Rot2& Ri,
                                       const Rot2& Rj,
                                       const Vector1& bias,
                                       OptionalMatrixType H1,
                                       OptionalMatrixType H2,
                                       OptionalMatrixType H3) const {
  return _PIM_.computeError(Ri, Rj, bias, H1, H2, H3);
}

//------------------------------------------------------------------------------
PlanarAHRSFactor::PlanarAHRSFactor(
    Key rot_i, Key rot_j, Key bias,
    const PreintegratedPlanarAhrsMeasurements& pim)
    : Base(noiseModel::Gaussian::Covariance(pim.preintMeasCov_), rot_i, rot_j,
           bias),
      _PIM_(pim) {
  auto p =
      std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>(pim.p());
  _PIM_.p_ = p;
}

//------------------------------------------------------------------------------
Rot2 PlanarAHRSFactor::predict(const Rot2& Ri, const Vector1& bias,
                               const PreintegratedPlanarAhrsMeasurements& pim) {
  auto p =
      std::make_shared<PreintegratedPlanarAhrsMeasurements::Params>(pim.p());
  PreintegratedPlanarAhrsMeasurements newPim = pim;
  newPim.p_ = p;
  return newPim.predict(Ri, bias);
}

}  // namespace gtsam