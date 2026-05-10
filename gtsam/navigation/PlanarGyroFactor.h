/**
 * A "between" factor for pose rotation, with variable bias.
 *
 * Useful for high-school robotics competitions,
 * which run robots on the floor, and so measure yaw.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarGyroFactor.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/navigation/PlanarGyroMeasurement.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>

#include "gtsam/dllexport.h"

namespace gtsam {
class GTSAM_EXPORT PlanarGyroFactor
    : public NoiseModelFactorN<Pose2, Pose2, double> {
  typedef PlanarGyroFactor This;
  typedef NoiseModelFactorN<Pose2, Pose2, double> Base;

  PlanarGyroMeasurement measurement_;

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /** Shorthand for a smart pointer to a factor */
#if !defined(_MSC_VER) && __GNUC__ == 4 && __GNUC_MINOR__ > 5
  typedef typename std::shared_ptr<PlanarGyroFactor> shared_ptr;
#else
  typedef std::shared_ptr<PlanarGyroFactor> shared_ptr;
#endif

  PlanarGyroFactor(Key pose_i, Key pose_j, Key bias,
                   const PlanarGyroMeasurement& measurement);

  ~PlanarGyroFactor() override {}

  gtsam::NonlinearFactor::shared_ptr clone() const override;
  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override;
  bool equals(const NonlinearFactor&, double tol = 1e-9) const override;

  /**
   * @param H1 dErr/dPi (3x3)
   * @param H2 dErr/dPj (3x3)
   * @param H3 dErr/dBias (3x1)
   * @return Vector3 err
   */
  Vector evaluateError(const Pose2& Pi, const Pose2& Pj, const double& bias,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3) const override;
};
}  // namespace gtsam