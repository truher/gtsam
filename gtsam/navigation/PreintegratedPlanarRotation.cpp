/**
 *  @file  PreintegratedPlanarRotation.cpp
 **/

#include "PreintegratedPlanarRotation.h"

using namespace std;

namespace gtsam {

void PreintegratedPlanarRotationParams::print(const string& s) const {
  cout << (s.empty() ? s : s + "\n") << endl;
  cout << "gyroscopeCovariance:\n" << gyroscopeCovariance << endl;
  if (body_P_sensor) body_P_sensor->print("body_P_sensor");
}

bool PreintegratedPlanarRotationParams::equals(
    const PreintegratedPlanarRotationParams& other, double tol) const {
  if (body_P_sensor) {
    if (!other.body_P_sensor ||
        !assert_equal(*body_P_sensor, *other.body_P_sensor, tol))
      return false;
  }
  return equal_with_abs_tol(gyroscopeCovariance, other.gyroscopeCovariance,
                            tol);
}

void PreintegratedPlanarRotation::resetIntegration() {
  deltaTij_ = 0.0;
  deltaRij_ = Rot2();
  delRdelBiasOmega_ = Z_3x3;
}

void PreintegratedPlanarRotation::print(const string& s) const {
  cout << s;
  cout << "    deltaTij [" << deltaTij_ << "]" << endl;
  cout << "    deltaRij.theta = (" << deltaRij_.theta() << ")" << endl;
}

bool PreintegratedPlanarRotation::equals(
    const PreintegratedPlanarRotation& other, double tol) const {
  return this->matchesParamsWith(other) &&
         deltaRij_.equals(other.deltaRij_, tol) &&
         std::abs(deltaTij_ - other.deltaTij_) < tol &&
         equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol);
}

namespace internal {
Rot2 IncrementalPlanarRotation::operator()(
    const Vector1& bias, OptionalJacobian<1, 1> H_bias) const {
  // First we compensate the measurements for the bias
  Vector3 correctedOmega = measuredOmega - bias;

  // Then compensate for sensor-body displacement: we express the quantities
  // (originally in the IMU frame) into the body frame.
  // Note that the rotate Jacobian is just body_P_sensor->rotation().matrix().
  if (body_P_sensor) {
    // rotation rate vector in the body frame
    correctedOmega = body_P_sensor->rotation() * correctedOmega;
  }

  // rotation vector describing rotation increment computed from the
  // current rotation rate measurement
  const Vector3 integratedOmega = correctedOmega * deltaT;
  Rot2 incrR = Rot2::Expmap(integratedOmega, H_bias);  // expensive !!
  if (H_bias) {
    *H_bias *= -deltaT;  // Correct so accurately reflects bias derivative
    if (body_P_sensor) *H_bias *= body_P_sensor->rotation().matrix();
  }
  return incrR;
}
}  // namespace internal

void PreintegratedPlanarRotation::integrateGyroMeasurement(
    const Vector1& measuredOmega, 
    const Vector1& biasHat,
    double deltaT,
    OptionalJacobian<1, 1> F) {
  Matrix3 H_bias;
  internal::IncrementalPlanarRotation f{measuredOmega, deltaT,
                                        p_->body_P_sensor};
  const Rot2 incrR = f(biasHat, H_bias);

  // Update deltaTij and rotation
  deltaTij_ += deltaT;
  deltaRij_ = deltaRij_.compose(incrR, F);

  // Update Jacobian
  const Matrix3 incrRt = incrR.transpose();
  delRdelBiasOmega_ = incrRt * delRdelBiasOmega_ + H_bias;
}


Rot2 PreintegratedPlanarRotation::biascorrectedDeltaRij(
    const Vector3& biasOmegaIncr, OptionalJacobian<1, 1> H) const {
  const Vector3 biasInducedOmega = delRdelBiasOmega_ * biasOmegaIncr;
  const Rot2 deltaRij_biascorrected = deltaRij_.expmap(biasInducedOmega, {}, H);
  if (H) (*H) *= delRdelBiasOmega_;
  return deltaRij_biascorrected;
}

}  // namespace gtsam
