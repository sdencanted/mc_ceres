#include "ceres/ceres.h"
#include "glog/logging.h"
#include <Eigen/Dense>

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
class CostFunctor
{
public:
  CostFunctor(
      const Eigen::Matrix<double, 3, 3> &camera_intrinsics)
      : camera_intrinsics_(camera_intrinsics)
  {
    camera_intrinsics_inv_ = camera_intrinsics_.inverse();
  }

  template <typename T>
  bool operator()(const T * x, const T *const  t, const T * rotations, T* residual) const
  {
    // Eigen::Matrix<T, 3, 3> rotation_matrix = (Eigen::AngleAxisf(rotations[0] * t[0], Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(rotations[1] * t[0], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(rotations[2]* t[0], Eigen::Vector3f::UnitZ())).normalized().toRotationMatrix().cast<T>();
    Eigen::Matrix<T, 3, 3> rotation_matrix =Eigen::Matrix<T, 3, 3>::Zero();
    rotation_matrix(0,1)=-rotations[2]*t[0];
    rotation_matrix(0,2)=rotations[1]*t[0];
    rotation_matrix(1,0)=rotations[2]*t[0];
    rotation_matrix(1,2)=-rotations[0]*t[0];
    rotation_matrix(2,0)=-rotations[1]*t[0];
    rotation_matrix(2,1)=rotations[0]*t[0];

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(x);

    const Eigen::Matrix<T, 3, 1> unprojected_point =
        (camera_intrinsics_inv_ * point);

    const Eigen::Matrix<T, 3, 1> rotated_point =
        (rotation_matrix * unprojected_point);

    const Eigen::Matrix<T, 2, 1> reprojected_pixel =
        (camera_intrinsics_ * rotated_point).hnormalized();
    residual[0] = reprojected_pixel[0] - reprojected_pixel[1];
    return true;
  }

  // // accepts u v and camera intrinsics flattened
  // // returns X Y Z
  // template <typename T>
  // T unproject(T coords,T intrinsics) const{
  //   T unprojected_coords[3];
  //   unprojected_coords[0]=(coords[0]/intrinsics[0])-intrinsics[2];
  //   unprojected_coords[1]=(coords[1]/intrinsics[4])-intrinsics[5];
  //   unprojected_coords[2]=1;
  //   return unprojected_coords;
  // }
  // // accepts X Y Z and camera intrinsics flattened
  // // returns u v
  // template <typename T>
  // T unproject(T coords,T intrinsics) const{
  //   T projected_coords[2];
  //   projected_coords[0]=(coords[0]/intrinsics[0])-intrinsics[2];
  //   projected_coords[1]=(coords[1]/intrinsics[4])-intrinsics[5];
  //   return projected_coords;
  // }

private:
  const Eigen::Matrix<double, 3, 3> &camera_intrinsics_;
  Eigen::Matrix<double, 3, 3> camera_intrinsics_inv_;
};
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x[3] = {1,66,77};
  double t = 0.5;
  double rotations[3] = {1, 1, 1};
  const double initial_x = x[0];
  // Build the problem.
  ceres::Problem problem;

  Eigen::Matrix<double, 3, 3> intrinsics=Eigen::Matrix<double, 3, 3>::Identity();
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  ceres::CostFunction *cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 3, 1, 3>(new CostFunctor(intrinsics));
  problem.AddResidualBlock(cost_function, nullptr, x, &t, rotations);
  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}