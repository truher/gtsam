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

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;


TEST(PlanarProjectionFactor, error) {
    Point3 landmark(1, 0, 0);
    Point2 measured(0, 0);
    Pose3 offset;
    Cal3DS2 calib(200, 200, 0, 200, 200, 0, 0);
    SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector2(1, 1));
    PlanarProjectionFactor factor(landmark, measured, offset, calib, model, X(0));
    Values values;
    Pose2 p0(0.05, 0, 0);
    Pose3 offset(Rot3(), Point3());

    Cal3DS2 KCAL(200.0, 200.0, 0.0, 200.0, 200.0, -0.2, 0.1);

    values.insert(X(0), 0);

    Vector expected = Vector4(0, 0, 0, 0);
    EXPECT(assert_equal(expected, factor.unwhitenedError(values)));

}