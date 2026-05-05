/**
 *  @file  PreintegratedPlanarRotation.cpp
 **/
#include "PreintegratedPlanarRotation.h"

using namespace std;

namespace gtsam {

void PreintegratedPlanarRotation::resetIntegration() {
  deltaTij_ = 0.0;
  deltaRij_ = Rot2();
  delRdelBiasOmega_ = Z_1x1;
}

void PreintegratedPlanarRotation::print(const string& s) const {
  cout << s;
  cout << "    deltaTij [" << deltaTij_ << "]" << endl;
  cout << "    deltaRij.theta = (" << deltaRij_.theta() << ")" << endl;
}

bool PreintegratedPlanarRotation::equals(
    const PreintegratedPlanarRotation& other, double tol) const {
  return std::abs(gyroscopeCovariance_ - other.gyroscopeCovariance_) < tol &&
         deltaRij_.equals(other.deltaRij_, tol) &&
         std::abs(deltaTij_ - other.deltaTij_) < tol &&
         equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol);
}

void PreintegratedPlanarRotation::integrateGyroMeasurement(double measuredOmega,
                                                           double biasHat,
                                                           double deltaT) {
  const Rot2 incrR = Rot2::fromAngle((measuredOmega - biasHat) * deltaT);


  // Update deltaTij and rotation
  deltaTij_ += deltaT;
  deltaRij_ = deltaRij_.compose(incrR);

  // Update Jacobian
  Matrix1 H_bias = I_1x1 * -deltaT;
  delRdelBiasOmega_ = delRdelBiasOmega_ + H_bias;
}

Rot2 PreintegratedPlanarRotation::biascorrectedDeltaRij(
    double biasOmegaIncr, OptionalJacobian<1, 1> H) const {
  const Vector1 biasInducedOmega = delRdelBiasOmega_ * biasOmegaIncr;
  const Rot2 deltaRij_biascorrected = deltaRij_.expmap(biasInducedOmega, {}, H);
  if (H) (*H) *= delRdelBiasOmega_;
  return deltaRij_biascorrected;
}
}  // namespace gtsam
