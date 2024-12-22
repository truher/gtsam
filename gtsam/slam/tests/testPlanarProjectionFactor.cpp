/**
 * @file testPlanarProjectionFactor.cpp
 * @date Dec 3, 2024
 * @author joel@truher.org
 * @brief unit tests for PlanarProjectionFactor
 */

#include <random>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PlanarProjectionFactor.h>
#include <gtsam/slam/ProjectionFactor.h>


#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::C;
using symbol_shorthand::K;
using symbol_shorthand::L;

// camera "zero" is facing +z; this turns it to face +x
// note (xy) parallels (uv) but this does not
const Pose3 CAM_COORD = Pose3(
    Rot3(0, 0, 1,//
        -1, 0, 0, //
        0, -1, 0),
    Vector3(0, 0, 0)
);

// this is the jacobian of the transform above
Matrix66 H00 = CAM_COORD.inverse().AdjointMap();

TEST(PlanarProjectionFactorBase, camera) {
    // the camera orientation using an "identity" pose3 
    // is facing +z, with (u,v) parallel to (x,y)
    PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(
        Pose3(),
        Cal3DS2(200, 200, 0, 200, 200, 0, 0));

    Matrix26 H1;
    Matrix23 H2;
    // +z => (u,v) on center
    CHECK(assert_equal(
        Vector2(200, 200),
        camera.project(Point3(0, 0, 1), H1, H2, {})));
    // "pan" (about y) => -u
    // "tilt" (about x) => v
    // "roll" (about z) => nothing
    // "truck" (along x) => -u
    // "pedestal" (along y) => -v
    // "dolly" (along z) => nothing 
    CHECK(assert_equal((Matrix26() << //
        0, -200, 0, -200, 0, 0,//
        200, 0, 0, 0, -200, 0).finished(), H1, 1e-6));
    // (u,v) is parallel to (x,y), and z never changes. 
    CHECK(assert_equal((Matrix23() << //
        200, 0, 0,//
        0, 200, 0).finished(), H2, 1e-6));

    // -y => (u,v) near the top of the camera frame
    // so the camera v axis is parallel with y
    CHECK(assert_equal(
        Vector2(200, 0),
        camera.project(Point3(0, -1, 1), H1, H2, {})));
    // "pan" (about y) => -u
    // "tilt" (about x) => v (fast)
    // "roll" (about z) => -u
    // "truck" (along x) => -u
    // "pedestal" (along y) => -v
    // "dolly" (along z) => -v 
    CHECK(assert_equal((Matrix26() << //
        0, -200, -200, -200, 0, 0,//
        400, 0, 0, 0, -200, -200).finished(), H1, 1e-6));
    // (u,v) is parallel to (x,y), point moving +z means +y. 
    CHECK(assert_equal((Matrix23() << //
        200, 0, 0,//
        0, 200, 200).finished(), H2, 1e-6));

    // -x => (u,v) near the left side of the camera frame
    // so the camera u axis is parallel with x
    CHECK(assert_equal(
        Vector2(0, 200),
        camera.project(Point3(-1, 0, 1), H1, H2, {})));
    // similar to above
    CHECK(assert_equal((Matrix26() << //
        0, -400, 0, -200, 0, -200,//
        200, 0, 200, 0, -200, 0).finished(), H1, 1e-6));
    // (u,v) is parallel to (x,y), point moving +z means -x. 
    CHECK(assert_equal((Matrix23() << //
        200, 0, 200,//
        0, 200, 0).finished(), H2, 1e-6));
}

TEST(PlanarProjectionFactorBase, camera2) {
    // to get a camera pointing down +x, with (u,v) antiparallel to (y,z),
    // we need a pose3 with this rotation
    PinholeCamera<Cal3DS2> camera = PinholeCamera<Cal3DS2>(
        CAM_COORD,
        Cal3DS2(200, 200, 0, 200, 200, 0, 0));

    // the camera z is parallel to world x, so it sees this:
    // on bore +x -> +z
    CHECK(assert_equal(Point3(0, 0, 1),
        camera.pose().transformTo(Point3(1, 0, 0))));
    // xz -> z-y
    CHECK(assert_equal(Point3(0, -1, 1),
        camera.pose().transformTo(Point3(1, 0, 1))));
    // xz -> z-y
    CHECK(assert_equal(Point3(-1, 0, 1),
        camera.pose().transformTo(Point3(1, 1, 0))));

    Matrix26 H1;
    Matrix23 H2;

    // +x => (u,v) on center
    CHECK(assert_equal(
        Vector2(200, 200),
        camera.project(Point3(1, 0, 0), H1, H2, {})));
    // because the camera has a non-identity pose, the jacobian
    // for camera motion is the same as the case above
    CHECK(assert_equal((Matrix26() << //
        0, -200, 0, -200, 0, 0,//
        200, 0, 0, 0, -200, 0).finished(), H1, 1e-6));
    // landmark movement in x makes no difference (it's on-bore)
    // +y means -u
    // +z means -v
    CHECK(assert_equal((Matrix23() << //
        0, -200, 0,//
        0, 0, -200).finished(), H2, 1e-6));

    // +z => (u,v) near the top of the camera frame
    // so the camera v axis is antiparallel with z
    CHECK(assert_equal(
        Vector2(200, 0),
        camera.project(Point3(1, 0, 1), H1, H2, {})));
    // camera motion is same as above
    CHECK(assert_equal((Matrix26() << //
        0, -200, -200, -200, 0, 0,//
        400, 0, 0, 0, -200, -200).finished(), H1, 1e-6));
    // landmark motion
    CHECK(assert_equal((Matrix23() << //
        0, -200, 0,//
        200, 0, -200).finished(), H2, 1e-6));

    // +y => (u,v) near the left side of the camera frame
    // so the camera u axis is antiparallel with y
    CHECK(assert_equal(
        Vector2(0, 200),
        camera.project(Point3(1, 1, 0), H1, H2, {})));
    // same as above
    CHECK(assert_equal((Matrix26() << //
        0, -400, 0, -200, 0, -200,//
        200, 0, 200, 0, -200, 0).finished(), H1, 1e-6));
    // landmark
    CHECK(assert_equal((Matrix23() << //
        200, -200, 0,//
        0, 0, -200).finished(), H2, 1e-6));
}

TEST(PlanarProjectionFactor1, error1) {
    // landmark is on the camera bore (which faces +x)
    Point3 landmark(1, 0, 0);
    // so pixel measurement is (cx, cy)
    Point2 measured(200, 200);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    Values values;
    Pose2 pose(0, 0, 0);
    values.insert(X(0), pose);

    PlanarProjectionFactor1 factor(
        X(0), landmark, measured, offset.compose(CAM_COORD), calib, model);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(1);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);

    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    const Matrix23 H1Expected = (Matrix23() << //
        0, 200, 200,//
        0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));
}

TEST(PlanarProjectionFactor1, error2) {
    // landmark is in the upper left corner
    Point3 landmark(1, 1, 1);
    // upper left corner in pixels
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor1 factor(
        X(0), landmark, measured, offset.compose(CAM_COORD), calib, model);
    Values values;
    Pose2 pose(0, 0, 0);

    values.insert(X(0), pose);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(1);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);

    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    Matrix23 H1Expected = (Matrix23() << //
        -200, 200, 400, //
        -200, 0, 200).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));
}

TEST(PlanarProjectionFactor1, error3) {
    // landmark is in the upper left corner
    Point3 landmark(1, 1, 1);
    // upper left corner in pixels
    Point2 measured(0, 0);
    Pose3 offset;
    // distortion
    Cal3DS2 calib(200, 200, 0, 200, 200, -0.2, 0.1);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor1 factor(
        X(0), landmark, measured, offset.compose(CAM_COORD), calib, model);
    Values values;
    Pose2 pose(0, 0, 0);

    values.insert(X(0), pose);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(1);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);

    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    Matrix23 H1Expected = (Matrix23() << //
        -360, 280, 640, //
        -360, 80, 440).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));
}

TEST(PlanarProjectionFactor1, jacobian) {
    // test many jacobians with many randoms

    std::default_random_engine rng(42);
    std::uniform_real_distribution<double> dist(-0.3, 0.3);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));

    for (int i = 0; i < 1000; ++i) {
        Point3 landmark(2 + dist(rng), dist(rng), dist(rng));
        Point2 measured(200 + 100 * dist(rng), 200 + 100 * dist(rng));
        Pose3 offset(Rot3::Ypr(dist(rng), dist(rng), dist(rng)), Point3(dist(rng), dist(rng), dist(rng)));
        Cal3DS2 calib(200, 200, 0, 200, 200, -0.2, 0.1);

        PlanarProjectionFactor1 factor(
            X(0), landmark, measured, offset.compose(CAM_COORD), calib, model);

        Pose2 pose(dist(rng), dist(rng), dist(rng));

        // actual H
        Matrix H1;
        factor.evaluateError(pose, H1);

        auto expectedH1 = numericalDerivative11<Vector, Pose2>(
            [&factor](const Pose2& p) {
                return factor.evaluateError(p, {});},
                pose);
        CHECK(assert_equal(expectedH1, H1, 1e-6));
    }
}

TEST(PlanarProjectionFactor1, solve) {
    // localization with covariance
    // constant landmark, offset, and calibration
    // two off-bore landmarks are enough.

    // offset here is identity, measured x-fwd
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    // the noise here is large in order to see the covariance
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(10, 10));


    // there's just one variable in our model, X(0).
    NonlinearFactorGraph graph;

    // these points are rather close together in order
    // to produce an observable covariance.
    // upper-left
    graph.add(PlanarProjectionFactor1(
        X(0),
        Point3(1, 0.1, 1),
        Point2(180, 0),
        offset.compose(CAM_COORD),
        calib,
        model));
    // upper-right
    graph.add(PlanarProjectionFactor1(
        X(0),
        Point3(1, -0.1, 1),
        Point2(220, 0),
        offset.compose(CAM_COORD),
        calib,
        model));
    // prior is a little bit wrong but also very soft
    graph.add(PriorFactor<Pose2>(
        X(0),
        Pose2(0.1, 0.1, 0.1),
        noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1))));
    // initial estimate is a little bit wrong
    Values initialEstimate;
    initialEstimate.insert(X(0), Pose2(0.1, 0.1, 0.1));

    // run the optimizer
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    // verify that the optimizer found the right pose.
    // note the somewhat high tolerance.
    Pose2 x0 = result.at<Pose2>(X(0));
    CHECK(assert_equal(Pose2(0, 0, 0), x0, 2e-3));

    // make sure the covariance is oriented correctly.
    // the "x" component variance is quite low because the landmarks
    // are far off-center vertically.
    // the "y" and "theta" components are much higher, and negatively
    // correlated: drift a bit to the left (+x) and you need to
    // rotate to the right (-theta) to stay on target.
    // note the somewhat high tolerance
    Marginals marginals(graph, result);
    Matrix cov = marginals.marginalCovariance(X(0));
    CHECK(assert_equal((Matrix33() << //
        0.001, 0, 0, //
        0, 0.1, -0.1, //
        0, -0.1, 0.1).finished(), cov, 3e-3));
}


TEST(PlanarProjectionFactor3, error1) {
    // landmark is on the camera bore (facing +x)
    Point3 landmark(1, 0, 0);
    // so px is (cx, cy)
    Point2 measured(200, 200);
    // offset is identity
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    Values values;
    Pose2 pose(0, 0, 0);
    values.insert(X(0), pose);
    values.insert(C(0), offset.compose(CAM_COORD));
    values.insert(K(0), calib);

    PlanarProjectionFactor3 factor(
        X(0), C(0), K(0), landmark, measured, model);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(3);

    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);
    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    // NOTE! composition of jacobian.
    // const Matrix& H2Actual = actualHs.at(1) * H00;
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    // du/dx etc for the pose2d
    Matrix23 H1Expected = (Matrix23() <<//
        0, 200, 200,//
        0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    // du/dx for the pose3d offset
    // note this is (roll, pitch, yaw, x, y, z)
    Matrix26 H2Expected = (Matrix26() <<//
        0, -200, 0, -200, 0, 0,//
        200, -0, 0, 0, -200, 0).finished();

    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    // du wrt calibration
    // on-bore, f doesn't matter
    // but c does
    Matrix29 H3Expected = (Matrix29() <<//
        0, 0, 0, 1, 0, 0, 0, 0, 0,//
        0, 0, 0, 0, 1, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H3Expected, H3Actual, 1e-6));
}

TEST(PlanarProjectionFactor3, error2) {
    Point3 landmark(1, 1, 1);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);

    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor3 factor(
        X(0), C(0), K(0), landmark, measured, model);
    Values values;
    Pose2 pose(0, 0, 0);

    values.insert(X(0), pose);
    values.insert(C(0), offset.compose(CAM_COORD));
    values.insert(K(0), calib);


    CHECK_EQUAL(2, model->dim());
    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);

    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    // NOTE! composition of jacobian.
    // const Matrix& H2Actual = actualHs.at(1) * H00;
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    Matrix23 H1Expected = (Matrix23() <<//
        -200, 200, 400,//
        -200, 0, 200).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    // du/dx for the pose3d offset
    // note this is (roll, pitch, yaw, x, y, z)
    Matrix26 H2Expected = (Matrix26() <<//
        200, -400, -200, -200, 0, -200,//
        400, -200, 200, 0, -200, -200).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    Matrix29 H3Expected = (Matrix29() <<//
        -1, 0, -1, 1, 0, -400, -800, 400, 800,//
        0, -1, 0, 0, 1, -400, -800, 800, 400).finished();

    CHECK(assert_equal(H3Expected, H3Actual, 1e-6));
}

TEST(PlanarProjectionFactor3, error3) {
    Point3 landmark(1, 1, 1);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, -0.2, 0.1);

    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor3 factor(
        X(0), C(0), K(0), landmark, measured, model);
    Values values;
    Pose2 pose(0, 0, 0);

    values.insert(X(0), pose);
    values.insert(C(0), offset.compose(CAM_COORD));
    values.insert(K(0), calib);


    CHECK_EQUAL(2, model->dim());
    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);

    CHECK(assert_equal(Vector2(0, 0), actual));

    const Matrix& H1Actual = actualHs.at(0);
    // NOTE! composition of jacobian.
    // const Matrix& H2Actual = actualHs.at(1) * H00;
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    Matrix23 H1Expected = (Matrix23() <<//
        -360, 280, 640,//
        -360, 80, 440).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    // du/dx for the pose3d offset
    // note this is (roll, pitch, yaw, x, y, z)
    Matrix26 H2Expected = (Matrix26() <<//
        440, -640, -200, -280, -80, -360,//
        640, -440, 200, -80, -280, -360).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    Matrix29 H3Expected = (Matrix29() <<//
        -1, 0, -1, 1, 0, -400, -800, 400, 800,//
        0, -1, 0, 0, 1, -400, -800, 800, 400).finished();

    CHECK(assert_equal(H3Expected, H3Actual, 1e-6));
}


TEST(PlanarProjectionFactor3, jacobian) {
    // test many jacobians with many randoms

    std::default_random_engine rng(42);
    std::uniform_real_distribution<double> dist(-0.3, 0.3);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));

    for (int i = 0; i < 1000; ++i) {
        Point3 landmark(2 + dist(rng), dist(rng), dist(rng));
        Point2 measured(200 + 100 * dist(rng), 200 + 100 * dist(rng));
        Pose3 offset = Pose3(
            Rot3::Ypr(dist(rng), dist(rng), dist(rng)),
            Point3(dist(rng), dist(rng), dist(rng))).compose(CAM_COORD);
        Cal3DS2 calib(200, 200, 0, 200, 200, -0.2, 0.1);

        PlanarProjectionFactor3 factor(
            X(0), C(0), K(0), landmark, measured, model);

        Pose2 pose(dist(rng), dist(rng), dist(rng));

        // actual H
        Matrix H1, H2, H3;
        factor.evaluateError(pose, offset, calib, H1, H2, H3);

        Matrix expectedH1 = numericalDerivative31<Vector, Pose2, Pose3, Cal3DS2>(
            [&factor](const Pose2& p, const Pose3& o, const Cal3DS2& c) {
                return factor.evaluateError(p, o, c, {}, {}, {});},
                pose, offset, calib);
        Matrix expectedH2 = numericalDerivative32<Vector, Pose2, Pose3, Cal3DS2>(
            [&factor](const Pose2& p, const Pose3& o, const Cal3DS2& c) {
                return factor.evaluateError(p, o, c, {}, {}, {});},
                pose, offset, calib);
        Matrix expectedH3 = numericalDerivative33<Vector, Pose2, Pose3, Cal3DS2>(
            [&factor](const Pose2& p, const Pose3& o, const Cal3DS2& c) {
                return factor.evaluateError(p, o, c, {}, {}, {});},
                pose, offset, calib);
        CHECK(assert_equal(expectedH1, H1, 1e-6));
        CHECK(assert_equal(expectedH2, H2, 1e-6));
        CHECK(assert_equal(expectedH3, H3, 1e-6));
    }
}

TEST(PlanarProjectionFactor3, solveOffset) {
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(
        Vector2(1, 1));

    Point3 l0(1, 0, 0);
    PlanarProjectionFactor3 factor1(
        X(0),
        C(0),
        K(0),
        l0,
        Point2(200, 200),
        model);
    Point3 l1(1, 0, -1);
    PlanarProjectionFactor3 factor2(
        X(0),
        C(0),
        K(0),
        l1,
        Point2(200, 400),
        model);
    Point3 l2(1, -1, 0);
    PlanarProjectionFactor3 factor3(
        X(0),
        C(0),
        K(0),
        l2,
        Point2(400, 200),
        model);
    Point3 l3(2, 2, 0);
    PlanarProjectionFactor3 factor4(
        X(0),
        C(0),
        K(0),
        l3,
        Point2(0, 200),
        model);
    Pose2 x0(0, 0, 0);
    Pose3 c0(
        Rot3(0, 0, 1, //
            -1, 0, 0, //
            0, -1, 0), //
        Vector3(0, 0, 0));
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);

    NonlinearFactorGraph graph;
    graph.add(factor1);
    graph.add(factor2);
    graph.add(factor3);
    graph.add(factor4);
    graph.add(PriorFactor<Pose2>(
        X(0),
        x0,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    // very loose prior for the offset
    graph.add(PriorFactor<Pose3>(
        C(0),
        c0,
        noiseModel::Diagonal::Sigmas(
            Vector6(10, 10, 10, 10, 10, 10))));
    graph.add(PriorFactor<Cal3DS2>(
        K(0),
        calib,
        noiseModel::Diagonal::Sigmas(
            Vector9(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))));

    Values initialEstimate;
    initialEstimate.insert(X(0), x0);
    initialEstimate.insert(C(0), c0);
    initialEstimate.insert(K(0), calib);

    // verify the error2 and jacobians
    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual1 = factor1.unwhitenedError(initialEstimate, actualHs);
    CHECK(assert_equal(Vector2(0, 0), actual1));

    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    // +y and +theta both => +u
    Matrix23 H1Expected = (Matrix23() <<//
        0, 200, 200,//
        0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    Matrix26 H2Expected = (Matrix26() << //
        000, -200, 000, -200, 000., 000, //
        200, 000., 000, 000., -200, 000).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    // jacobian for the calibration
    // on-bore, the only thing that matters is cx, cy.
    Matrix29 H3Expected = (Matrix29() <<//
        0, 0, 0, 1, 0, 0, 0, 0, 0,//
        0, 0, 0, 0, 1, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H3Expected, H3Actual, 1e-6));

    // run the optimizer
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    // verify that the optimizer found the right pose.
    // note the somewhat high tolerance.
    Pose2 xx0 = result.at<Pose2>(X(0));
    CHECK(assert_equal(Pose2(0, 0, 0), xx0, 2e-3));

    // verify the camera is pointing at +x
    Pose3 cc0 = result.at<Pose3>(C(0));
    CHECK(assert_equal(Pose3(
        Rot3(0, 0, 1,//
            -1, 0, 0, //
            0, -1, 0),
        Vector3(0, 0, 0)), cc0, 5e-3));

    // to get to the x-forward offset we expect, invert the transform
    Pose3 c0Xfwd = cc0.compose(CAM_COORD.inverse());
    CHECK(assert_equal(Pose3(), c0Xfwd, 2e-3));

    // verify the calibration
    Cal3DS2 k0 = result.at<Cal3DS2>(K(0));
    CHECK(assert_equal(Cal3DS2(200, 200, 0, 200, 200, 0, 0), k0, 2e-3));

    Marginals marginals(graph, result);

    Matrix x0cov = marginals.marginalCovariance(X(0));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,//
        0, 0, 0,//
        0, 0, 0).finished(), x0cov, 1e-4));

    Matrix c0cov = marginals.marginalCovariance(C(0));
    CHECK(assert_equal((Matrix66() << //
        8.30128e-05, 9.10201e-06, -2.36966e-05, -2.63024e-05, 0.000105938, -2.87623e-05,
        9.10201e-06, 0.000116649, -5.37453e-06, -2.48266e-05, 1.17142e-05, -4.42248e-06,
        -2.36966e-05, -5.37453e-06, 1.69549e-05, 1.48553e-05, -3.19123e-05, 9.13905e-06,
        -2.63024e-05, -2.48266e-05, 1.48553e-05, 0.00015081, -3.35284e-05, 1.45632e-05,
        0.000105938, 1.17142e-05, -3.19123e-05, -3.35284e-05, 0.000145587, -3.32703e-05,
        -2.87623e-05, -4.42248e-06, 9.13905e-06, 1.45632e-05, -3.32703e-05, 0.000127789).finished(),
        c0cov, 1e-4));

    // inverse adjoint of the camera offset
    Matrix66 ccH00 = cc0.inverse().AdjointMap();

    // invert the camera offset: covariance in body coordinates
    Matrix c0cov2 = ccH00.inverse() * c0cov * ccH00.inverse().transpose();

    // these are the **camera-frame** std deviations
    Vector6 c0sigma = c0cov.diagonal().cwiseSqrt();
    CHECK(assert_equal((Vector6() << //
        0.00911113,
        0.0108004,
        0.00411764,
        0.0122805,
        0.0120659,
        0.0113044
        ).finished(), c0sigma, 1e-6));

    // this is **body frame**
    Vector6 bTcSigma = c0cov2.diagonal().cwiseSqrt();
    CHECK(assert_equal((Vector6() << //
        0.00411764,
        0.00911113,
        0.0108004,
        0.0113044,
        0.0122805,
        0.0120659
        ).finished(), bTcSigma, 1e-6));

    // calibration prior means this is roughly zero
    Matrix k0cov = marginals.marginalCovariance(K(0));
    CHECK(assert_equal((Matrix99() << //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0).finished(), k0cov, 3e-3));
}

TEST(PlanarProjectionFactor3, solveWithoutOffset) {
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(10, 10));
    NonlinearFactorGraph graph;

    Point3 l0(0, 0, 1);
    PlanarProjectionFactor3 factor1(
        X(0),
        C(0),
        K(0),
        l0,
        Point2(200, 200),
        model);
    graph.add(factor1);
    Point3 l1(0, 1, 1);
    PlanarProjectionFactor3 factor2(
        X(0),
        C(0),
        K(0),
        l1,
        Point2(200, 400),
        model);
    graph.add(factor2);
    Point3 l2(1, 0, 1);
    PlanarProjectionFactor3 factor3(
        X(0),
        C(0),
        K(0),
        l2,
        Point2(400, 200),
        model);
    graph.add(factor3);

    Pose2 x0(0, 0, 0);
    graph.add(PriorFactor<Pose2>(
        X(0),
        x0,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    Pose3 c0(
        Rot3(1, 0, 0, //
            0, 1, 0, //
            0, 0, 1), //
        Vector3(0, 0, 0));
    graph.add(PriorFactor<Pose3>(
        C(0),
        c0,
        noiseModel::Diagonal::Sigmas(
            Vector6(100, 100, 100, 100, 100, 100))));
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    graph.add(PriorFactor<Cal3DS2>(
        K(0),
        calib,
        noiseModel::Diagonal::Sigmas(
            Vector9(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))));

    Values initialEstimate;
    initialEstimate.insert(X(0), x0);
    initialEstimate.insert(C(0), c0);
    initialEstimate.insert(L(0), l0);
    initialEstimate.insert(L(1), l1);
    initialEstimate.insert(L(2), l2);
    initialEstimate.insert(K(0), calib);

    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual1 = factor1.unwhitenedError(initialEstimate, actualHs);
    CHECK(assert_equal(Vector2(0, 0), actual1));

    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    Matrix23 H1Expected = (Matrix23() <<//
        -200, 0, 0,//
        0, -200, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    Matrix26 H2Expected = (Matrix26() << //
        0, -200, 0, -200, 0, 0,//
        200, 0, 0, 0, -200, 0).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    Matrix29 H3Expected = (Matrix29() <<//
        0, 0, 0, 1, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 1, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H3Expected, H3Actual, 1e-6));

    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    Pose2 xx0 = result.at<Pose2>(X(0));
    CHECK(assert_equal(Pose2(0, 0, 0), xx0, 2e-3));

    // offset should be identity
    Pose3 cc0 = result.at<Pose3>(C(0));
    CHECK(assert_equal(Pose3(
        Rot3(1, 0, 0,//
            0, 1, 0, //
            0, 0, 1),
        Vector3(0, 0, 0)), cc0, 5e-6));

    // calib should be the initial value
    Cal3DS2 k0 = result.at<Cal3DS2>(K(0));
    CHECK(assert_equal(Cal3DS2(200, 200, 0, 200, 200, 0, 0), k0, 2e-3));

    Marginals marginals(graph, result);

    // tight prior => low cov
    Matrix x0cov = marginals.marginalCovariance(X(0));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,//
        0, 0, 0,//
        0, 0, 0).finished(), x0cov, 1e-4));

    // this is the covariance in the camera frame, i.e.
    // z-forward, y-down.  note the very low roll variance,
    // and the high variance for everything else, caused
    // by the prior.
    Matrix c0cov = marginals.marginalCovariance(C(0));
    CHECK(assert_equal((Matrix66() << //
        2000, -2000, 0, 2000, 2000, -2000,//
        -2000, 2000, 0, -2000, -2000, 2000,//
        0, 0, 0, 0, 0, 0,//
        2000, -2000, 0, 2000, 2000, -2000,//
        2000, -2000, 0, 2000, 2000, -2000,//
        -2000, 2000, 0, -2000, -2000, 2000).finished(),
        c0cov, 2));

    // tight prior => low cov
    Matrix k0cov = marginals.marginalCovariance(K(0));
    CHECK(assert_equal((Matrix99() << //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0).finished(), k0cov, 3e-3));
}


TEST(PlanarProjectionFactor3, compareToGeneric) {
    // compare to the normal projection factor.
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(10, 10));
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    auto K = std::make_shared<Cal3DS2>(200, 200, 0, 200, 200, 0, 0);

    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor1(
        Point2(200, 200),
        model,
        C(0),
        L(0), // landmark, should be Point3(0, 0, 1)
        K);

    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor2(
        Point2(200, 400),
        model,
        C(0),
        L(1), // landmark, should be Point3(0, 1, 1)
        K);
    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor3(
        Point2(400, 200),
        model,
        C(0),
        L(2), // landmark, should be Point3(1, 0, 1)
        K);

    Values initialEstimate;
    Pose3 c0(
        Rot3(1, 0, 0, //
            0, 1, 0, //
            0, 0, 1), //
        Vector3(0, 0, 0));
    initialEstimate.insert(C(0), c0);
    Point3 l0(0, 0, 1);
    initialEstimate.insert(L(0), l0);
    Point3 l1(0, 1, 1);
    initialEstimate.insert(L(1), l1);
    Point3 l2(1, 0, 1);
    initialEstimate.insert(L(2), l2);

    Matrix H1;
    Matrix H2;
    Vector e0 = factor1.evaluateError(c0, l0, &H1, &H2);
    CHECK(assert_equal((Matrix26() << //
        0, -200, 0, -200, 0, 0,
        200, -0, 0, 0, -200, 0).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        200, 0, 0,//
        0, 200, 0).finished(), H2, 1e-6));
    CHECK(assert_equal(Vector2(0, 0), e0, 1e-6));

    Vector e1 = factor2.evaluateError(c0, l1, &H1, &H2);
    CHECK(assert_equal(Vector2(0, 0), e1, 1e-6));
    CHECK(assert_equal((Matrix26() << //
        0, -200, 200, -200, 0, 0,
        400, -0, 0, 0, -200, 200).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        200, 0, 0,//
        0, 200, -200).finished(), H2, 1e-6));

    Vector e2 = factor3.evaluateError(c0, l2, &H1, &H2);
    CHECK(assert_equal(Vector2(0, 0), e2, 1e-6));
    CHECK(assert_equal((Matrix26() << //
        0, -400, 0, -200, 0, 200,
        200, -0, -200, 0, -200, 0).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        200, -0, -200,//
        0, 200, 0).finished(), H2, 1e-6));


    std::vector<Matrix> actualHs(2);
    gtsam::Vector actual1 = factor1.unwhitenedError(initialEstimate, actualHs);
    CHECK(assert_equal(Vector2(0, 0), actual1));

    //
    // check jacobians
    //
    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);

    Matrix26 H1Expected = (Matrix26() <<//
        0, -200, 0, -200, 0, 0, //
        200, 0, 0, 0, -200, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    Matrix23 H2Expected = (Matrix23() << //
        200, 0, 0,//
        0, 200, 0).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    NonlinearFactorGraph graph;
    graph.add(factor1);
    graph.add(factor2);
    graph.add(factor3);
    graph.add(PriorFactor<Pose3>(
        C(0),
        Pose3(),
        noiseModel::Diagonal::Sigmas(Vector6(1, 1, 1, 1, 1, 1))));
    graph.add(PriorFactor<Point3>(
        L(0),
        Point3(0, 0, 1),
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    graph.add(PriorFactor<Point3>(
        L(1),
        Point3(0, 1, 1),
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    graph.add(PriorFactor<Point3>(
        L(2),
        Point3(1, 0, 1),
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));

    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    Pose3 cc0 = result.at<Pose3>(C(0));
    CHECK(assert_equal(Pose3(
        Rot3(1, 0, 0,//
            0, 1, 0, //
            0, 0, 1),
        Vector3(0, 0, 0)), cc0, 1e-9));

    Point3 ll0 = result.at<Point3>(L(0));
    CHECK(assert_equal(Point3(0, 0, 1), ll0, 1e-9));
    Point3 ll1 = result.at<Point3>(L(1));
    CHECK(assert_equal(Point3(0, 1, 1), ll1, 1e-9));
    Point3 ll2 = result.at<Point3>(L(2));
    CHECK(assert_equal(Point3(1, 0, 1), ll2, 1e-9));

    //
    // check covariance
    //
    Marginals marginals(graph, result);

    // this is exactly the same as my own example
    // x not coupled with yz
    // y inverse with z, which is correct.
    // rx coupled with y, which is correct
    // rx inverse with z, which is wrong ... ?
    // ry inverse with x which is correct
    // rz not coupled with anything
    Matrix c0cov = marginals.marginalCovariance(C(0));
    CHECK(assert_equal((Matrix66() << //
        0.20, -.20, -.00, 0.20, 0.20, -.20,
        -.20, 0.20, -.00, -.20, -.20, 0.20,
        -.00, 0.00, 0.00, 0.00, -.00, 0.00,
        0.20, -.20, 0.00, 0.21, 0.20, -.20,
        0.20, -.20, -.00, 0.20, 0.21, -.20,
        -.20, 0.20, 0.00, -.20, -.20, 0.20
        ).finished(),
        c0cov, 0.01));

    Matrix l0cov = marginals.marginalCovariance(L(0));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,
        0, 0, 0,
        0, 0, 0).finished(),
        l0cov, 0.01));

    Matrix l1cov = marginals.marginalCovariance(L(1));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,
        0, 0, 0,
        0, 0, 0).finished(),
        l1cov, 0.01));
}


TEST(PlanarProjectionFactor3, compareToGeneric2) {
    // normal projection with camera in correct orientation
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    shared_ptr<Cal3DS2> K = std::make_shared<Cal3DS2>(200, 200, 0, 200, 200, 0, 0);


    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor1(
        Point2(200, 200),
        model,
        C(0),
        L(0),
        K);
    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor2(
        Point2(200, 400),
        model,
        C(0),
        L(1),
        K);
    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor3(
        Point2(400, 200),
        model,
        C(0),
        L(2),
        K);
    GenericProjectionFactor<Pose3, Point3, Cal3DS2> factor4(
        Point2(0, 200),
        model,
        C(0),
        L(3),
        K);

    Values initialEstimate;
    Pose3 c0(
        Rot3(0, 0, 1, //
            -1, 0, 0, //
            0, -1, 0), //
        Vector3(0, 0, 0));
    initialEstimate.insert(C(0), c0);
    Point3 l0(1, 0, 0);
    initialEstimate.insert(L(0), l0);
    Point3 l1(1, 0, -1);
    initialEstimate.insert(L(1), l1);
    Point3 l2(1, -1, 0);
    initialEstimate.insert(L(2), l2);
    // this improves accuracy a lot
    Point3 l3(2, 2, 0);
    initialEstimate.insert(L(3), l3);

    // the errors are all zero at the initial pose
    Matrix H1;
    Matrix H2;
    Vector e0 = factor1.evaluateError(c0, l0, &H1, &H2);
    CHECK(assert_equal((Matrix26() << //
        0, -200, 0, -200, 0, 0,
        200, -0, 0, 0, -200, 0).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        0, -200, 0,//
        0, 0, -200).finished(), H2, 1e-6));
    CHECK(assert_equal(Vector2(0, 0), e0, 1e-6));

    Vector e1 = factor2.evaluateError(c0, l1, &H1, &H2);
    CHECK(assert_equal(Vector2(0, 0), e1, 1e-6));
    CHECK(assert_equal((Matrix26() << //
        0, -200, 200, -200, 0, 0,
        400, -0, 0, 0, -200, 200).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        0, -200, 0, //
        -200, 0, -200).finished(), H2, 1e-6));

    Vector e2 = factor3.evaluateError(c0, l2, &H1, &H2);
    CHECK(assert_equal(Vector2(0, 0), e2, 1e-6));
    CHECK(assert_equal((Matrix26() << //
        0, -400, 0, -200, 0, 200,
        200, -0, -200, 0, -200, 0).finished(), H1, 1e-6));
    CHECK(assert_equal((Matrix23() << //
        -200, -200, 0,//
        0, 0, -200).finished(), H2, 1e-6));

    std::vector<Matrix> actualHs(2);
    gtsam::Vector actual1 = factor1.unwhitenedError(initialEstimate, actualHs);
    CHECK(assert_equal(Vector2(0, 0), actual1, 1e-6));

    //
    // check jacobians
    //

    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);

    // pose jacobian
    Matrix26 H1Expected = (Matrix26() <<//
        0, -200, 0, -200, 0, 0, //
        200, 0, 0, 0, -200, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-6));

    // point jacobian
    Matrix23 H2Expected = (Matrix23() << //
        0, -200, 0,//
        0, 0, -200).finished();
    CHECK(assert_equal(H2Expected, H2Actual, 1e-6));

    NonlinearFactorGraph graph;
    graph.add(factor1);
    graph.add(factor2);
    graph.add(factor3);
    graph.add(factor4);

    // *very* soft prior, so that the solver finds the right answer.
    // a better estimate also works.
    graph.add(PriorFactor<Pose3>(
        C(0),
        c0,
        noiseModel::Diagonal::Sigmas(Vector6(10, 10, 10, 10, 10, 10))));

    // landmark priors are very tight
    graph.add(PriorFactor<Point3>(
        L(0),
        l0,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    graph.add(PriorFactor<Point3>(
        L(1),
        l1,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    graph.add(PriorFactor<Point3>(
        L(2),
        l2,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));
    graph.add(PriorFactor<Point3>(
        L(3),
        l3,
        noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01))));

    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    Pose3 cc0 = result.at<Pose3>(C(0));
    CHECK(assert_equal(c0, cc0, 0.05));
    Point3 ll0 = result.at<Point3>(L(0));
    CHECK(assert_equal(l0, ll0, 1e-4));
    Point3 ll1 = result.at<Point3>(L(1));
    CHECK(assert_equal(l1, ll1, 1e-4));
    Point3 ll2 = result.at<Point3>(L(2));
    CHECK(assert_equal(l2, ll2, 1e-4));
    Point3 ll3 = result.at<Point3>(L(3));
    CHECK(assert_equal(l3, ll3, 1e-4));
    //
    // check covariance
    //

    Marginals marginals(graph, result);

    Matrix c0cov = marginals.marginalCovariance(C(0));
    CHECK(assert_equal((Matrix66() << //
        0.000467927, 6.60775e-05, -0.000143849, -0.000162094, 0.00060235, -0.000160038,
        6.60775e-05, 5.59977e-05, -2.77214e-05, -0.000101612, 7.99381e-05, -4.85882e-05,
        -0.000143849, -2.77214e-05, 7.56042e-05, 7.25797e-05, -0.00018165, 5.52411e-05,
        -0.000162094, -0.000101612, 7.25797e-05, 0.000234676, -0.000198384, 0.000104031,
        0.00060235, 7.99381e-05, -0.00018165, -0.000198384, 0.000818174, -0.000187658,
        -0.000160038, -4.85882e-05, 5.52411e-05, 0.000104031, -0.000187658, 0.000157096).finished(),
        c0cov, 1e-4));

    // inverse adjoint of the camera offset
    Matrix66 ccH00 = cc0.inverse().AdjointMap();

    // invert the camera offset: covariance in body coordinates
    Matrix c0cov2 = ccH00.inverse() * c0cov * ccH00.inverse().transpose();

    // these are the **camera-frame** std deviations
    Vector6 c0sigma = c0cov.diagonal().cwiseSqrt();
    CHECK(assert_equal((Vector6() << //
        0.0216316, // pitch
        0.00748316, // yaw
        0.00869506, // roll
        0.0153191, // camera x is world y
        0.0286037, // camera y is world z
        0.0125338 // camera z is world x
        ).finished(), c0sigma, 1e-6));

    // this is **body frame**
    Vector6 bTcSigma = c0cov2.diagonal().cwiseSqrt();
    CHECK(assert_equal((Vector6() << //
        0.00869506,
        0.0216316,
        0.00748316,
        0.0125338,
        0.0153191,
        0.0286037
        ).finished(), bTcSigma, 1e-6));


    Matrix l0cov = marginals.marginalCovariance(L(0));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,
        0, 0, 0,
        0, 0, 0).finished(),
        l0cov, 0.01));
    Matrix l1cov = marginals.marginalCovariance(L(1));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,
        0, 0, 0,
        0, 0, 0).finished(),
        l1cov, 0.01));
    Matrix l2cov = marginals.marginalCovariance(L(2));
    CHECK(assert_equal((Matrix33() << //
        0, 0, 0,
        0, 0, 0,
        0, 0, 0).finished(),
        l2cov, 0.01));

}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */

