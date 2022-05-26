#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ctime>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char **argv) {
  // generate matrix
  Matrix<float, 2, 3> matrix_23;

  Vector3d v_3d;
  Matrix<float, 3, 1> vd_3d;

  Matrix3d matrix_33 = Matrix3d::Zero();

  Matrix<double, Dynamic, Dynamic> matrix_dynamic;

  MatrixXd matrix_x;

  // matrix operation

  matrix_23 << 1, 2, 3, 4, 5, 6;
  cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

  cout << "second row of matrix 2x3: \n" << matrix_23.row(1) << endl;

  cout << "print matrix 2x3: " << endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
    cout << endl;
  }

  v_3d << 1, 2, 3;
  vd_3d << 4, 5, 6;

  // wrong case
  // Matrix<float, 2, 3> result = matrix_23 * v_3d.cast<float>();

  Matrix<float, 2, 1> result = matrix_23 * v_3d.cast<float>();
  cout << "[1,2,3;4,5,6]*[1,2,3]=" << result.transpose() << endl;

  // other operations
  cout
      << "************************few matrix operations************************"
      << endl;
  srand(2022);
  matrix_33 = Matrix3d::Random();
  cout << "random matrix: \n" << matrix_33 << endl;
  cout << "transpose: \n" << matrix_33.transpose() << endl;
  cout << "sum: " << matrix_33.sum() << endl;
  cout << "trace: " << matrix_33.trace() << endl;
  cout << "times 10: \n" << 10 * matrix_33 << endl;
  cout << "inverse: \n" << matrix_33.inverse() << endl;
  cout << "check inverser, expected identity: \n"
       << matrix_33 * matrix_33.inverse() << endl;
  cout << "det: " << matrix_33.determinant() << endl;

  cout << "************************eigen values************************"
       << endl;
  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() *
                                                matrix_33);
  cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
  cout << "Eigen vectors: \n" << eigen_solver.eigenvectors() << endl;

  cout << "************************solve equation************************"
       << endl;
  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN =
      MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN.transpose() * matrix_NN;
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_start = clock();

  Matrix<double, MATRIX_SIZE, 1> x_brute_force = matrix_NN.inverse() * v_Nd;
  cout << "time of brute force inverse is: "
       << 1000 * (clock() - time_start) / (double)CLOCKS_PER_SEC << "ms"
       << endl;
  cout << "x = " << x_brute_force.transpose() << endl;

  time_start = clock();
  Matrix<double, MATRIX_SIZE, 1> x_QR =
      matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time of QR decomposition is: "
       << 1000 * (clock() - time_start) / (double)CLOCKS_PER_SEC << "ms"
       << endl;
  cout << "x = " << x_QR.transpose() << endl;

  time_start = clock();
  Matrix<double, MATRIX_SIZE, 1> x_cholesky = matrix_NN.ldlt().solve(v_Nd);
  cout << "time of ldlt is: "
       << 1000 * (clock() - time_start) / (double)CLOCKS_PER_SEC << "ms"
       << endl;
  cout << "x = " << x_cholesky.transpose() << endl;

  return 0;
}