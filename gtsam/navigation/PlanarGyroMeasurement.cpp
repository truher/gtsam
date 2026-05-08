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
  cout << " dt [" << deltaT_ << "]" << endl;
  cout << " dtheta = (" << deltaR_.theta() << ")" << endl;
}

bool PlanarGyroMeasurement::equals(const PlanarGyroMeasurement& other,
                                   double tol) const {
  return std::abs(ARW_ - other.ARW_) < tol &&
         deltaR_.equals(other.deltaR_, tol) &&
         std::abs(deltaT_ - other.deltaT_) < tol;
}

Rot2 PlanarGyroMeasurement::biascorrectedDeltaR(
    double bias, OptionalJacobian<1, 1> H) const {
  const double dtheta = -deltaT_ * bias;
  const Rot2 deltaRij_biascorrected = deltaR_.compose(Rot2::fromAngle(dtheta));
  // bias derivative is just opposite of the time
  if (H) (*H)(0) = -deltaT_;
  return deltaRij_biascorrected;
}

void PlanarGyroMeasurement::integrateMeasurement(double omega,
                                                 double deltaT) {
  const Rot2 dtheta = Rot2::fromAngle(omega * deltaT);
  deltaT_ += deltaT;
  deltaR_ = deltaR_.compose(dtheta);
}

Rot2 PlanarGyroMeasurement::predict(const Rot2& Ri, double bias,
                                    gtsam::OptionalJacobian<1, 1> H1,
                                    gtsam::OptionalJacobian<1, 1> H2) const {
  const Rot2 biascorrected = this->biascorrectedDeltaR(bias, H2);
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