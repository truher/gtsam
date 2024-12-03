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

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::C;
using symbol_shorthand::K;

TEST(PlanarSFMFactor, error) {
    PlanarSFMFactor factor();
    Values values;
    Pose2 p0(0.05, 0, 0);
    Pose3 offset(Rot3(), Point3());

    Cal3DS2 KCAL(200.0, 200.0, 0.0, 200.0, 200.0, -0.2, 0.1);

    values.insert(X(0), p0);
    values.insert(C(0), offset);
    values.insert(K(0), KCAL);

    Vector expected = Vector4(0, 0, 0, 0);
    EXPECT(assert_equal(expected, factor.unwhitenedError(values)));

}