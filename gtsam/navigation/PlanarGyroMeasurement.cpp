/**
 * @file PlanarGyroMeasurement.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */
#include <gtsam/navigation/PlanarGyroMeasurement.h>

#include <iostream>

namespace gtsam {
void PlanarGyroMeasurement::print(const std::string& s) const {
  std::cout << s;
  std::cout << " dt [" << deltaT_ << "]" << std::endl;
  std::cout << " dtheta = (" << deltaR_.theta() << ")" << std::endl;
}

bool PlanarGyroMeasurement::equals(const PlanarGyroMeasurement& other,
                                   double tol) const {
  return std::abs(ARW_ - other.ARW_) < tol &&
         deltaR_.equals(other.deltaR_, tol) &&
         std::abs(deltaT_ - other.deltaT_) < tol;
}

Rot2 PlanarGyroMeasurement::deltaR(double bias,
                                   OptionalJacobian<1, 1> H) const {
  if (H) (*H)(0) = -deltaT_;
  return deltaR_.compose(Rot2::fromAngle(-deltaT_ * bias));
}

Rot2 PlanarGyroMeasurement::predict(const Rot2& Ri, double bias,
                                    OptionalJacobian<1, 1> H1,
                                    OptionalJacobian<1, 1> H2) const {
  return Ri.compose(deltaR(bias, H2), H1);
}

double PlanarGyroMeasurement::computeError(const Rot2& Ri, const Rot2& Rj,
                                           double bias,
                                           OptionalJacobian<1, 1> H1,
                                           OptionalJacobian<1, 1> H2,
                                           OptionalJacobian<1, 1> H3) const {
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

  return error(0);
}
}  // namespace gtsam