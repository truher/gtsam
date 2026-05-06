/**
 * @file PlanarGyro.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */
#include <gtsam/navigation/PlanarGyro.h>

#include <iostream>

using namespace std;

namespace gtsam {
void PlanarGyro::print(const string& s) const {
  cout << s;
  cout << " deltaTij [" << deltaTij_ << "]" << endl;
  cout << " eltaRij.theta = (" << deltaRij_.theta() << ")" << endl;
  cout << " biasHat [" << biasHat_ << "]" << endl;
  cout << " PreintMeasCov [ " << preintMeasCov_ << " ]" << endl;
}

bool PlanarGyro::equals(const PlanarGyro& other, double tol) const {
  return std::abs(gyroscopeCovariance_ - other.gyroscopeCovariance_) < tol &&
         deltaRij_.equals(other.deltaRij_, tol) &&
         std::abs(deltaTij_ - other.deltaTij_) < tol &&
         equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol) &&
         abs(biasHat_ - other.biasHat_) < tol;
}

void PlanarGyro::resetIntegration() {
  deltaTij_ = 0.0;
  deltaRij_ = Rot2();
  delRdelBiasOmega_ = Z_1x1;
  preintMeasCov_.setZero();
}

void PlanarGyro::integrateGyroMeasurement(double measuredOmega, double biasHat,
                                          double deltaT) {
  const Rot2 incrR = Rot2::fromAngle((measuredOmega - biasHat) * deltaT);
  deltaTij_ += deltaT;
  deltaRij_ = deltaRij_.compose(incrR);
  Matrix1 H_bias = I_1x1 * -deltaT;
  delRdelBiasOmega_ = delRdelBiasOmega_ + H_bias;
}

Rot2 PlanarGyro::biascorrectedDeltaRij(double biasOmegaIncr,
                                       OptionalJacobian<1, 1> H) const {
  const Vector1 biasInducedOmega = delRdelBiasOmega_ * biasOmegaIncr;
  const Rot2 deltaRij_biascorrected = deltaRij_.expmap(biasInducedOmega, {}, H);
  if (H) (*H) *= delRdelBiasOmega_;
  return deltaRij_biascorrected;
}

void PlanarGyro::integrateMeasurement(double measuredOmega, double deltaT) {
  integrateGyroMeasurement(measuredOmega, biasHat_, deltaT);
  Matrix1 newNoise{gyroscopeCovariance_ * deltaT};
  preintMeasCov_ += newNoise;
}

Rot2 PlanarGyro::predict(const Rot2& Ri, double bias,
                         gtsam::OptionalJacobian<1, 1> H1,
                         gtsam::OptionalJacobian<1, 1> H2) const {
  double biasOmegaIncr = bias - biasHat_;
  const Rot2 biascorrected = this->biascorrectedDeltaRij(biasOmegaIncr, H2);
  return Ri.compose(biascorrected, H1);
}

Vector1 PlanarGyro::computeError(const Rot2& Ri, const Rot2& Rj, double bias,
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
}  // namespace gtsam