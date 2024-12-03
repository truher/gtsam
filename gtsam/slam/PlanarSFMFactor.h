/**
 * Similar to GeneralSFMFactor, but:
 *
 * * 2d pose variable (robot on the floor)
 * * camera offset variable
 * * constant landmarks
 * * batched input
 * * numeric differentiation
 *
 * @file PlanarSFMFactor.h
 * @brief for planar smoothing with unknown calibration
 * @date Dec 2, 2024
 * @author joel@truher.org
 */
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
namespace gtsam {
    class PlanarSFMFactor : public NoiseModelFactor {
    public:
        PlanarSFMFactor() {}
        ~PlanarSFMFactor() override {}
        Vector evaluateError(const Pose3& pose, const Point3& point,
            OptionalMatrixType H1, OptionalMatrixType H2) const override {
        }
    };
} // namespace gtsam