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
 * The python version of this factor uses batches, to save on calls
 * across the C++/python boundary, but here the only extra cost
 * is instantiating the camera, so there's no batch.
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

#include <gtsam/base/Testable.h>
#include <gtsam/base/Lie.h>

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>


namespace gtsam {
    /**
     * @class PlanarSFMFactor
     * @brief Camera calibration for robot on the floor.
     */
    class PlanarSFMFactor : public NoiseModelFactorN<Pose2, Pose3, Cal3DS2> {
        static const Pose3 CAM_COORD;

    protected:

        Point3 landmark_; // landmark
        Point2 measured_; // pixel measurement

    public:
        PlanarSFMFactor() {}
        /**
         * @param landmarks point in the world
         * @param measured corresponding point in the camera frame
         * @param model stddev of the measurements, ~one pixel?
         * @param poseKey index of the robot pose2 in the z=0 plane
         * @param offsetKey index of the 3d camera offset from the robot pose
         * @param calibKey index of the camera calibration
         */
        PlanarSFMFactor(
            const Point3& landmark,
            const Point2& measured,
            const SharedNoiseModel& model,
            Key poseKey,
            Key offsetKey,
            Key calibKey)
            : NoiseModelFactorN(model, poseKey, offsetKey, calibKey),
            landmark_(landmark), measured_(measured)
        {
            assert(2 == model->dim());
        }

        ~PlanarSFMFactor() override {}

            /// @return a deep copy of this factor
        gtsam::NonlinearFactor::shared_ptr clone() const override {
            return std::static_pointer_cast<gtsam::NonlinearFactor>(
                gtsam::NonlinearFactor::shared_ptr(new PlanarSFMFactor(*this))); }


        Point2 h(const Pose2& pose,
            const Pose3& offset,
            const Cal3DS2& calib) const {
            // this is x-forward z-up
            Pose3 offset_pose = Pose3(pose).compose(offset);
            // this is z-forward y-down
            Pose3 camera_pose = offset_pose.compose(CAM_COORD);
            PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(camera_pose, calib);
            camera.project2(landmark_) - measured_;
        }

        Vector evaluateError(
            const Pose2& pose,
            const Pose3& offset,
            const Cal3DS2& calib,
            OptionalMatrixType H1 = OptionalNone,
            OptionalMatrixType H2 = OptionalNone,
            OptionalMatrixType H3 = OptionalNone
        ) const override {
            try {
                Point2 result = h(pose, offset, calib) - measured_;
                if (H1) *H1 = numericalDerivative31<Point2, Pose2, Pose3, Cal3DS2>(
                    [&](const Pose2& p, const Pose3& o, const Cal3DS2& c) {return h(p, o, c);},
                    pose, offset, calib);
                if (H2) *H2 = numericalDerivative32<Point2, Pose2, Pose3, Cal3DS2>(
                    [&](const Pose2& p, const Pose3& o, const Cal3DS2& c) {return h(p, o, c);},
                    pose, offset, calib);
                if (H3) *H3 = numericalDerivative33<Point2, Pose2, Pose3, Cal3DS2>(
                    [&](const Pose2& p, const Pose3& o, const Cal3DS2& c) {return h(p, o, c);},
                    pose, offset, calib);
                return result;
            }
            catch (CheiralityException& e) {
                // TODO: what should these sizes be?
                if (H1) *H1 = Matrix::Zero(2, 3);
                if (H2) *H2 = Matrix::Zero(2, 6);
                if (H3) *H3 = Matrix::Zero(2, 9);
                // we don't know what to return here so return zero.
                // TODO: maybe return a big number instead?
                return Matrix::Zero(2, 1);
            }
        }
    };

    // camera "zero" is facing +z; this turns it to face +x
    const Pose3 PlanarSFMFactor::CAM_COORD = Pose3(
        Rot3(0, 0, 1,//
            -1, 0, 0, //
            0, -1, 0),
        Vector3(0, 0, 0)
    );

    template<>
    struct traits<PlanarSFMFactor> :
        public Testable<PlanarSFMFactor > {
    };

} // namespace gtsam