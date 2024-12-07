/**
 * @file testPlanarProjectionFactor.cpp
 * @date Dec 3, 2024
 * @author joel@truher.org
 * @brief unit tests for PlanarProjectionFactor
 */

#include <gtsam/slam/PlanarProjectionFactor.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/Testable.h>

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;


TEST(PlanarProjectionFactor, error1) {
    Point3 landmark(1, 0, 0);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor factor(landmark, measured, offset, calib, model, X(0));
    Values values;
    Pose2 pose(0.05, 0, 0);

    values.insert(X(0), pose);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(1);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);
    EQUALITY(Vector2(0, 0), actual);
    const Matrix& H1Actual = actualHs.at(0);
    Matrix23 H1Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-9));
}

TEST(PlanarProjectionFactor, error2) {
    Point3 landmark(1, 1, 1);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor factor(landmark, measured, offset, calib, model, X(0));
    Values values;
    Pose2 pose(0.05, 0, 0);

    values.insert(X(0), pose);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(1);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);
    EQUALITY(Vector2(0, 0), actual);
    const Matrix& H1Actual = actualHs.at(0);
    Matrix23 H1Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-9));
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */

