/**
 * Similar to GeneralSFMFactor, but:
 *
 * * 2d pose variable (robot on the floor)
 * * camera offset variable
 * * constant landmarks
 * * batched input
 * * numeric differentiation
 *
 * This factor is useful to find camera calibration and placement, in
 * a sort of "autocalibrate" mode.  Once a satisfactory solution is
 * found, the PlanarProjectionFactor should be used for localization.
 *
 * @see https://www.firstinspires.org/
 * @see PlanarProjectionFactor.h
 *
 * @file PlanarSFMFactor.h
 * @brief for planar smoothing with unknown calibration
 * @date Dec 2, 2024
 * @author joel@truher.org
 */
#pragma once

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {
    // camera "zero" is facing +z; this turns it to face +x
    const Pose3 PlanarSFMFactor::CAM_COORD = Pose3(
        Rot3(0, 0, 1,//
            -1, 0, 0, //
            0, -1, 0),
        Vector3(0, 0, 0)
    );

    class PlanarSFMFactor : public NoiseModelFactorN<Pose2, Pose3, Cal3DS2> {
        static const int pose2dim = FixedDimension<Pose2>::value;
        static const int pose3dim = FixedDimension<Pose3>::value;
        static const int calDim = FixedDimension<Cal3DS2>::value;
        static const Pose3 CAM_COORD;


    protected:

        std::list<Point3> landmarks_; // batch of landmarks
        std::list<Point2> measured_; // batch of pixel measurements

    public:
        /**
         * @param landmarks points in the world
         * @param measured corresponding points in the camera frame
         * @param model stddev of the measurements, ~one pixel?
         * @param poseKey index of the robot pose2 in the z=0 plane
         * @param offsetKey index of the 3d camera offset from the robot pose
         * @param calibKey index of the camera calibration
         */
        PlanarSFMFactor(
            const std::list<Point3>& landmarks,
            const std::list<Point2>& measured,
            const SharedNoiseModel& model,
            Key poseKey,
            Key offsetKey,
            Key calibKey)
            : NoiseModelFactorN(model, poseKey, offsetKey, calibKey),
            landmarks_(landmarks), measured_(measured)
        {
        }

        ~PlanarSFMFactor() override {}

        Vector evaluateError(
            const Pose2& pose,
            const Pose3& offset,
            const Cal3DS2& calib,
            OptionalMatrixType H1,
            OptionalMatrixType H2,
            OptionalMatrixType H3
        ) const override {
            try {
                // this is x-forward z-up
                Pose3 offset_pose = Pose3(pose).compose(offset);
                // this is z-forward y-down
                Pose3 camera_pose = offset_pose.compose(CAM_COORD);
                PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(camera_pose, calib);
                camera.project2(point, H1, H2) - measured_;

            }
            catch (CheiralityException& e) {
                // TODO: what should these sizes be?
                if (H1) *H1 = Matrix::Zero(2 * measured_.size(), pose2dim);
                if (H2) *H2 = Matrix::Zero(2 * measured_.size(), pose3dim);
                if (H3) *H3 = Matrix::Zero(2 * measured_.size(), calDim);
                // we don't know what to return here so return zero.
                // TODO: maybe return a big number instead?
                return Matrix::Zero(2 * measured_.size(), 1);
            }
        }
    };
} // namespace gtsam