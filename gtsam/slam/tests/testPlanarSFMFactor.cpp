/**
 * @file testPlanarSFMFactor.cpp
 * @date Dec 3, 2024
 * @author joel@truher.org
 * @brief unit tests for PlanarSFMFactor
 */

#include <gtsam/slam/PlanarSFMFactor.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/Testable.h>

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::C;
using symbol_shorthand::K;

TEST(PlanarSFMFactor, error1) {
    std::cout << "wtf\n";
    Point3 landmark(1, 0, 0);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);

    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarSFMFactor factor(landmark, measured, model, X(0), C(0), K(0));
    Values values;
    Pose2 pose(0.05, 0, 0);

    values.insert(X(0), pose);
    values.insert(C(0), offset);
    values.insert(K(0), calib);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);
    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);
    EQUALITY(Vector2(0, 0), actual);
    Matrix23 H1Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    Matrix23 H2Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    Matrix23 H3Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-9));
    CHECK(assert_equal(H2Expected, H2Actual, 1e-9));
    CHECK(assert_equal(H3Expected, H3Actual, 1e-9));
}

TEST(PlanarSFMFactor, error2) {
    std::cout << "wtf2\n";
    Point3 landmark(1, 1, 1);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);

    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarSFMFactor factor(landmark, measured, model, X(0), C(0), K(0));
    Values values;
    Pose2 pose(0.05, 0, 0);

    values.insert(X(0), pose);
    values.insert(C(0), offset);
    values.insert(K(0), calib);

    CHECK_EQUAL(2, factor.dim());
    CHECK(factor.active(values));
    std::vector<Matrix> actualHs(3);
    gtsam::Vector actual = factor.unwhitenedError(values, actualHs);
    const Matrix& H1Actual = actualHs.at(0);
    const Matrix& H2Actual = actualHs.at(1);
    const Matrix& H3Actual = actualHs.at(2);

    std::cout << H1Actual;

    EQUALITY(Vector2(0, 0), actual);
    Matrix13 H1Expected = (Matrix13() << 0, 0, 0).finished();
    Matrix23 H2Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    Matrix23 H3Expected = (Matrix23() << 0, 0, 0, 0, 0, 0).finished();
    CHECK(assert_equal(H1Expected, H1Actual, 1e-9));
    CHECK(assert_equal(H2Expected, H2Actual, 1e-9));
    CHECK(assert_equal(H3Expected, H3Actual, 1e-9));
}

TEST(PlanarSFMFactor, error3) {
    CHECK(2 == 3);
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */

