/**
 * @file PlanarGyroFactor.cpp
 * @author joel@truher.org
 * @date May 1, 2026
 */
#include <gtsam/navigation/PlanarGyroFactor.h>

#include <iostream>

using namespace std;

namespace gtsam {

PlanarGyroFactor::PlanarGyroFactor(Key pose_i, Key pose_j, Key bias,
                                   const PlanarGyroMeasurement& measurement)
    : Base(noiseModel::Diagonal::Variances(
               Vector3(1e-9, 1e-9, measurement.variance())),
           pose_i, pose_j, bias),
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
  return e != nullptr && Base::equals(*e, tol) &&
         measurement_.equals(e->measurement_, tol);
}

Vector PlanarGyroFactor::evaluateError(const Pose2& Pi, const Pose2& Pj,
                                       const double& bias,
                                       OptionalMatrixType H1,
                                       OptionalMatrixType H2,
                                       OptionalMatrixType H3) const {
  Matrix1 rH1;
  Matrix1 rH2;
  Matrix1 rH3;

  double err = measurement_.computeError(Pi.r(), Pj.r(), bias, H1 ? &rH1 : 0,
                                         H2 ? &rH2 : 0, H3 ? &rH3 : 0);
  // the only derivative is the theta one.
  if (H1) {
    *H1 = Z_3x3;
    H1->block<1, 1>(2, 2) = rH1;
  }
  if (H2) {
    *H2 = Z_3x3;
    H2->block<1, 1>(2, 2) = rH2;
  }
  if (H3) {
    *H3 = Z_3x1;
    H3->block<1, 1>(2, 0) = rH3;
  }
  // the only error is the theta one
  return Vector3(0, 0, err);
}
}  // namespace gtsam