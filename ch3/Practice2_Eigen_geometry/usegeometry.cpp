#include <cmath>
#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc, char **argv) {
  Matrix3d rotation_matrix = Matrix3d::Identity();

  AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));
  cout.precision(3);
  cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;

  rotation_matrix = rotation_vector.toRotationMatrix();

  Vector3d v(1, 0, 0);
  Vector3d v_rotated = rotation_vector * v;

  cout << "(1,0,0) after rotation by angle axis = " << v_rotated.transpose()
       << endl;

  cout << "(1,0,0) after rotation by matrix = "
       << (rotation_matrix * v).transpose() << endl;

  Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
  cout << "yaw pitch roll is:" << euler_angles.transpose() << endl;

  Isometry3d T = Isometry3d::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(Vector3d(1, 3, 5));
  cout << "transform matrix = \n" << T.matrix() << endl;

  Vector3d v_transformed = T * v;
  cout << "v_transformed =" << v_transformed.transpose() << endl;

  Quaterniond q = Quaterniond(rotation_matrix);
  cout << "quaternion from rotation matrix = " << q.coeffs().transpose()
       << endl;

  cout << "quaternion from rotation vector = "
       << Quaterniond(rotation_vector).coeffs().transpose() << endl;

  cout << "(1,0,0) after rotation by quaternion = " << (q * v).transpose()
       << endl;

  cout << "shoule be equal to: "
       << (q * Quaterniond(0, 1, 0, 0) * q.inverse())
              .coeffs()
              .block(0, 0, 3, 1)
              .transpose()
       << endl;
  return 0;
}