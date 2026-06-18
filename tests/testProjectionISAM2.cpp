#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::L;
using symbol_shorthand::X;

static SharedNoiseModel model(noiseModel::Unit::Create(2));
static Cal3DS2::shared_ptr K(new Cal3DS2(200.0, 200.0, 0.0, 400.0, 300.0, 0.0,
                                         0.0));

Values initial() {
  Values initial;
  initial.insert(X(0), Pose3(Rot3::Ypr(0, 0, 0), Point3(0, 0, 0)));
  initial.insert(L(0), Point3(0, 0, 0));
  initial.insert(L(1), Point3(0, 0, 0));
  initial.insert(L(2), Point3(0, 0, 0));
  initial.insert(L(3), Point3(0, 0, 0));
  return initial;
}
NonlinearFactorGraph graph() {
  NonlinearFactorGraph graph;

  // priors
  auto poseNoise = noiseModel::Isotropic::Sigma(6, 100);
  auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.001);
  graph.push_back(PriorFactor<Pose3>(X(0), Pose3(), poseNoise));
  graph.push_back(
      PriorFactor<Point3>(L(0), Point3(8.0, 4.25, 0.75), pointNoise));
  graph.push_back(
      PriorFactor<Point3>(L(1), Point3(8.0, 3.75, 0.75), pointNoise));
  graph.push_back(
      PriorFactor<Point3>(L(2), Point3(8.0, 3.75, 1.25), pointNoise));
  graph.push_back(
      PriorFactor<Point3>(L(3), Point3(8.0, 4.25, 1.25), pointNoise));

  Pose3 offset(Rot3(0, 0, 1,   //
                    -1, 0, 0,  //
                    0, -1, 0),
               Vector3(0, 0, 0.5));
  // measurements
  graph.push_back(GenericProjectionFactor<Pose3, Point3, Cal3DS2>(
      Point2(325, 291.6667), model, X(0), L(0), K, offset));
  graph.push_back(GenericProjectionFactor<Pose3, Point3, Cal3DS2>(
      Point2(341.6667, 291.6667), model, X(0), L(1), K, offset));
  graph.push_back(GenericProjectionFactor<Pose3, Point3, Cal3DS2>(
      Point2(341.6667, 275), model, X(0), L(2), K, offset));
  graph.push_back(GenericProjectionFactor<Pose3, Point3, Cal3DS2>(
      Point2(325, 275), model, X(0), L(3), K, offset));
  return graph;
}

TEST(testProjectionISAM2, projectionGN) {
  cout << " ********** PROJECTION GAUSS NEWTON **********" << endl;
  GaussNewtonOptimizer optimizer(graph(), initial());
  Values result = optimizer.optimize();
  result.print("result");
}

TEST(testProjectionISAM2, projectionISAM) {
  cout << " ********** PROJECTION ISAM2 **********" << endl;
  ISAM2Params parameters;
  // relinearize every time, helps a lot!
  parameters.relinearizeSkip = 1;
  // these seem not to help
  // parameters.relinearizeThreshold = 0.01;
  // parameters.factorization = ISAM2Params::Factorization::QR;
  // parameters.optimizationParams = ISAM2DoglegParams(
  // 1.0, 1e-5, DoglegOptimizerImpl::SEARCH_EACH_ITERATION, true);
  // parameters.optimizationParams = ISAM2GaussNewtonParams(1.0);
  ISAM2 isam2(parameters);
  ISAM2Result isam2Result = isam2.update(graph(), initial());
  isam2Result.print();
  Values result = isam2.calculateBestEstimate();
  result.print("result");

  isam2Result = isam2.update();
  isam2Result.print();
  result = isam2.calculateBestEstimate();
  result.print("result");

  isam2Result = isam2.update();
  isam2Result.print();
  result = isam2.calculateBestEstimate();
  result.print("result");

  isam2Result = isam2.update();
  isam2Result.print();
  result = isam2.calculateBestEstimate();
  result.print("result");

  isam2Result = isam2.update();
  isam2Result.print();
  result = isam2.calculateBestEstimate();
  result.print("result");
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
