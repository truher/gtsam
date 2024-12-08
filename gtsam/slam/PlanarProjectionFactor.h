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
 * The python version of this factor uses batches, to save on calls
 * across the C++/python boundary, but here the only extra cost
 * is instantiating the camera, so there's no batch.
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
     * @class PlanarProjectionFactor
     * @brief Camera projection for robot on the floor.
    */
    class PlanarProjectionFactor : public NoiseModelFactorN<Pose2> {
        static const Pose3 CAM_COORD;

    protected:

        Point3 landmark_; // landmark
        Point2 measured_; // pixel measurement
        Pose3 offset_; // camera offset to robot pose
        Cal3DS2 calib_; // camera calibration

    public:
        PlanarProjectionFactor(){}

        /**
         * @param landmarks point in the world
         * @param measured corresponding point in the camera frame
         * @param offset constant camera offset from pose
         * @param calib constant camera calibration
         * @param model stddev of the measurements, ~one pixel?
         * @param poseKey index of the robot pose in the z=0 plane
         */
        PlanarProjectionFactor(
            const Point3& landmark,
            const Point2& measured,
            const Pose3& offset,
            const Cal3DS2& calib,
            const SharedNoiseModel& model,
            Key poseKey)
            : NoiseModelFactorN(model, poseKey),
            landmark_(landmark),
            measured_(measured),
            offset_(offset),
            calib_(calib)
        {
            assert(2 == model->dim());
        }

        ~PlanarProjectionFactor() override {}

        /// @return a deep copy of this factor
        gtsam::NonlinearFactor::shared_ptr clone() const override {
            return std::static_pointer_cast<gtsam::NonlinearFactor>(
                gtsam::NonlinearFactor::shared_ptr(new PlanarProjectionFactor(*this))); }


        Point2 h(const Pose2& pose) const {
            // this is x-forward z-up
            Pose3 offset_pose = Pose3(pose).compose(offset_);
            // this is z-forward y-down
            Pose3 camera_pose = offset_pose.compose(CAM_COORD);
            PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(camera_pose, calib_);
            return camera.project2(landmark_);
        }

        Vector evaluateError(
            const Pose2& pose,
            OptionalMatrixType H1 = OptionalNone) const override {
            try {
                Point2 result = h(pose) - measured_;
                if (H1) *H1 = numericalDerivative11<Point2, Pose2>(
                    [&](const Pose2& p) {return h(p);},
                    pose);
                return result;
            }
            catch (CheiralityException& e) {
                // TODO: check the size here
                if (H1) *H1 = Matrix::Zero(2, 3);
                // return a large error
                return Matrix::Constant(2, 1, 2.0 * calib_.fx());
            }
        }
    };

    // camera "zero" is facing +z; this turns it to face +x
    const Pose3 PlanarProjectionFactor::CAM_COORD = Pose3(
        Rot3(0, 0, 1,//
            -1, 0, 0, //
            0, -1, 0),
        Vector3(0, 0, 0)
    );

    template<>
    struct traits<PlanarProjectionFactor> :
        public Testable<PlanarProjectionFactor > {
    };

} // namespace gtsam