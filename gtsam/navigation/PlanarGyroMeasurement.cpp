/**
 * @file PlanarGyroMeasurement.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */
#include <gtsam/navigation/PlanarGyroMeasurement.h>

#include <iostream>

using namespace std;

namespace gtsam {
void PlanarGyroMeasurement::print(const string& s) const {
  cout << s;
  cout << " dt [" << deltaTij_ << "]" << endl;
  cout << " dtheta = (" << deltaRij_.theta() << ")" << endl;
}

bool PlanarGyroMeasurement::equals(const PlanarGyroMeasurement& other,
                                   double tol) const {
  return std::abs(gyroscopeCovariance_ - other.gyroscopeCovariance_) < tol &&
         deltaRij_.equals(other.deltaRij_, tol) &&
         std::abs(deltaTij_ - other.deltaTij_) < tol;
}

Rot2 PlanarGyroMeasurement::biascorrectedDeltaRij(
    double biasOmegaIncr, OptionalJacobian<1, 1> H) const {
  const double biasInducedOmega = -deltaTij_ * biasOmegaIncr;
  const Rot2 deltaRij_biascorrected = deltaRij_.compose(Rot2::fromAngle(biasInducedOmega));
  // bias derivative is just opposite of the time
  if (H) (*H)(0) = -deltaTij_;
  return deltaRij_biascorrected;
}

void PlanarGyroMeasurement::integrateMeasurement(double measuredOmega,
                                                 double deltaT) {
  const Rot2 incrR = Rot2::fromAngle(measuredOmega * deltaT);
  deltaTij_ += deltaT;
  deltaRij_ = deltaRij_.compose(incrR);
}

Rot2 PlanarGyroMeasurement::predict(const Rot2& Ri, double bias,
                                    gtsam::OptionalJacobian<1, 1> H1,
                                    gtsam::OptionalJacobian<1, 1> H2) const {
  double biasOmegaIncr = bias;
  const Rot2 biascorrected = this->biascorrectedDeltaRij(biasOmegaIncr, H2);
  return Ri.compose(biascorrected, H1);
}

Vector1 PlanarGyroMeasurement::computeError(
    const Rot2& Ri, const Rot2& Rj, double bias,
    gtsam::OptionalJacobian<1, 1> H1, gtsam::OptionalJacobian<1, 1> H2,
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