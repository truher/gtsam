/**
 * Similar to GenericProjectionFactor, but:
 *
 * * 2d pose variable (robot on the floor)
 * * constant landmarks
 * * batched input
 * * numeric differentiation
 *
 * @file PlanarSFMFactor.h
 * @brief for planar smoothing
 * @date Dec 2, 2024
 * @author joel@truher.org
 */
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>
namespace gtsam {
    class PlanarProjectionFactor : public NoiseModelFactor {
    public:
        PlanarProjectionFactor() {}
        ~PlanarProjectionFactor() override {}
        Vector evaluateError(const Pose3& pose, const Point3& point,
            OptionalMatrixType H1, OptionalMatrixType H2) const override {
        }

    };
} // namespace gtsam