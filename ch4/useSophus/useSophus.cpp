#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  cout << "****************Lie group and Lie algebra***************" << endl;
  Matrix3d rotation_z_90 =
      AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  Quaterniond q(rotation_z_90);

  Sophus::SO3d SO3_R(rotation_z_90);
  Sophus::SO3d SO3_q(q);

  cout << "SO(3) from rotation matrix is: \n" << SO3_R.matrix() << endl;
  cout << "SO(3) from quaternion is: \n" << SO3_q.matrix() << endl;

  Vector3d so3 = SO3_R.log();
  cout << "so3 =" << so3.transpose() << endl;
  cout << "so3 hat = \n" << Sophus::SO3d::hat(so3) << endl;
  cout << "so3 hat vee = "
       << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  cout << "*******************************" << endl;
  Vector3d update_so3(1e-4, 0, 0);
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated =  \n" << SO3_updated.matrix() << endl;

  cout << "**************SE(3)**********" << endl;
  Vector3d t(1, 0, 0);
  Sophus::SE3d SE3(rotation_z_90, t);
  cout << "SE3 = \n" << SE3.matrix() << endl;

  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3.log();

  cout << "se3 = " << se3.transpose() << endl;
  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)) << endl;

  Vector6d update_se3;
  update_se3.setZero();
  update_se3(0, 0) = 1e-4;

  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3;
  cout << "SE3 updated = \n" << SE3_updated.matrix() << endl;

  return 0;
}