/**
 * Like AHRSFactor, except:
 *
 * * using Rot2 instead of Rot3
 * * without the preintegrator: measurements are preintegrated by the gyro
 * hardware
 * * without coriolis correction: rotating reference frame is irrelevant
 * * without deprecated v4 stuff
 * * without the body transform: translation doesn't matter, rotation is always
 * identity.
 *
 * This factor is useful for high-school robotics competitions,
 * which run robots on the floor: they really only care about yaw.
 *
 * @see https://www.firstinspires.org/
 *
 * @file PlanarGyroFactor.h
 * @author joel@truher.org
 * @date May 1, 2026
 */

#pragma once
#include <gtsam/navigation/PlanarGyro.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/std_optional_serialization.h>
#include <gtsam/geometry/Rot2.h>

#include "gtsam/dllexport.h"

#include <optional>

namespace gtsam {

class GTSAM_EXPORT PlanarGyroFactor
    : public NoiseModelFactorN<Rot2, Rot2, double> {
  typedef PlanarGyroFactor This;
  typedef NoiseModelFactorN<Rot2, Rot2, double> Base;

  PlanarGyro measurement_;

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  /** Shorthand for a smart pointer to a factor */
#if !defined(_MSC_VER) && __GNUC__ == 4 && __GNUC_MINOR__ > 5
  typedef typename std::shared_ptr<PlanarGyroFactor> shared_ptr;
#else
  typedef std::shared_ptr<PlanarGyroFactor> shared_ptr;
#endif

  PlanarGyroFactor(Key rot_i, Key rot_j, Key bias,
                   const PlanarGyro& measurement);

  ~PlanarGyroFactor() override {}

  gtsam::NonlinearFactor::shared_ptr clone() const override;

  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override;

  bool equals(const NonlinearFactor&, double tol = 1e-9) const override;

  Vector evaluateError(const Rot2& Ri, const Rot2& Rj, const double& bias,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3) const override;
};
}  // namespace gtsam