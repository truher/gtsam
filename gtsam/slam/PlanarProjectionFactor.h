/**
 * Similar to GenericProjectionFactor, but:
 *
 * * 2d pose variable (robot on the floor)
 * * constant landmarks
 * * batched input
 * * numeric differentiation
 *
 * This factor is useful for high-school robotics competitions,
 * which run robots on the floor, with use fixed maps and fiducial
 * markers.  The camera offset and calibration are fixed, perhaps
 * found using PlanarSFMFactor.
 *
 * @see https://www.firstinspires.org/
 * @see PlanarSFMFactor.h
 *
 * @file PlanarSFMFactor.h
 * @brief for planar smoothing
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
    const Pose3 PlanarProjectionFactor::CAM_COORD = Pose3(
        Rot3(0, 0, 1,//
            -1, 0, 0, //
            0, -1, 0),
        Vector3(0, 0, 0)
    );

    class PlanarProjectionFactor : public NoiseModelFactorN<Pose2> {
        static const int pose2dim = FixedDimension<Pose2>::value;
        static const Pose3 CAM_COORD;

    protected:

        std::list<Point3> landmarks_; // batch of landmarks
        std::list<Point2> measured_; // batch of pixel measurements
        Pose3 offset_; // camera offset to robot pose
        Cal3DS2 calib_; // camera calibration

    public:
        /**
         * @param landmarks points in the world
         * @param measured corresponding points in the camera frame
         * @param offset constant camera offset from pose
         * @param calib constant camera calibration
         * @param model stddev of the measurements, ~one pixel?
         * @param poseKey index of the robot pose in the z=0 plane
         */
        PlanarProjectionFactor(
            const std::list<Point3>& landmarks,
            const std::list<Point2>& measured,
            const Pose3& offset,
            const Cal3DS2& calib,
            const SharedNoiseModel& model,
            Key poseKey)
            : NoiseModelFactorN(model, poseKey),
            landmarks_(landmarks),
            measured_(measured),
            offset_(offset),
            calib_(calib)
        {
        }

        ~PlanarProjectionFactor() override {}

        Vector evaluateError(
            const Pose2& pose,
            OptionalMatrixType H1) const override {
            try {
                // this is x-forward z-up
                Pose3 offset_pose = Pose3(pose).compose(offset);
                // this is z-forward y-down
                Pose3 camera_pose = offset_pose.compose(CAM_COORD);
                PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(camera_pose, calib);
                camera.project2(point, H1, H2) - measured_;
            }
            catch (CheiralityException& e) {
                // TODO: check the size here
                if (H1) *H1 = Matrix::Zero(2 * measured_.size(), pose2dim);
                // return a large error
                return Matrix::Constant(2 * measured_.size(), 1, 2.0 * calib_.fx());
            }
        }

    };
} // namespace gtsam