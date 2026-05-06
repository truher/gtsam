/**
 * @file PlanarGyroFactor.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */
#include <gtsam/navigation/PlanarGyroFactor.h>

#include <iostream>

using namespace std;

namespace gtsam {

PlanarGyroFactor::PlanarGyroFactor(Key rot_i, Key rot_j, Key bias,
                                   const PlanarGyro& measurement)
    : Base(noiseModel::Gaussian::Covariance(measurement.preintMeasCov_), rot_i, rot_j,
           bias),
      measurement_(measurement) {}

gtsam::NonlinearFactor::shared_ptr PlanarGyroFactor::clone() const {
  return std::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
}

void PlanarGyroFactor::print(const string& s,
                             const KeyFormatter& keyFormatter) const {
  cout << s << "PlanarGyroFactor(" << keyFormatter(this->key<1>()) << ","
       << keyFormatter(this->key<2>()) << "," << keyFormatter(this->key<3>())
       << ",";
  measurement_.print("  preintegrated measurements:");
  noiseModel_->print("  noise model: ");
}

bool PlanarGyroFactor::equals(const NonlinearFactor& other, double tol) const {
  const This* e = dynamic_cast<const This*>(&other);
  return e != nullptr && Base::equals(*e, tol) && measurement_.equals(e->measurement_, tol);
}

Vector PlanarGyroFactor::evaluateError(const Rot2& Ri, const Rot2& Rj,
                                       const double& bias,
                                       OptionalMatrixType H1,
                                       OptionalMatrixType H2,
                                       OptionalMatrixType H3) const {
  return measurement_.computeError(Ri, Rj, bias, H1, H2, H3);
}
}  // namespace gtsam