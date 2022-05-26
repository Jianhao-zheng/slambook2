#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  Quaterniond q1(0.35, 0.2, 0.3, 0.1);
  q1.normalize();
  Vector3d t1(0.3, 0.1, 0.1);
  Isometry3d T_R1W = Isometry3d::Identity();
  T_R1W.rotate(q1);
  T_R1W.pretranslate(t1);

  Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
  q2.normalize();
  Vector3d t2(-0.1, 0.5, 0.3);
  Isometry3d T_R2W = Isometry3d::Identity();
  T_R2W.rotate(q2);
  T_R2W.pretranslate(t2);

  Vector3d P_R1(0.5, 0, 0.2);
  Vector3d P_R2 = T_R2W * T_R1W.inverse() * P_R1;

  cout << "coordinate of point in R2 frame is:" << P_R2.transpose() << endl;

  return 0;
}