//
// Created by xiang on 18-11-19.
//

#include <ceres/ceres.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  template <typename T>
  bool operator()(const T *const abc, T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) +
                                     abc[2]);  // y-exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y;
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;   // real value
  double ae = 2.0, be = -1.0, ce = 5.0;  // estimated value

  double sigma = 1.0;
  double inv_sigma = 1.0 / sigma;

  int N = 100;
  cv::RNG rng;

  vector<double> x_sample, y_sample;
  x_sample.resize(N);
  y_sample.resize(N);
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    // this generator can estimate the exact value
    //  double x = rng.uniform((double)-5, (double)5);
    x_sample[i] = x;
    y_sample[i] = exp(ar * x * x + br * x + cr) + rng.gaussian(sigma * sigma);
  }

  double abc[3] = {ae, be, ce};

  // generate the problem structure
  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
            new CURVE_FITTING_COST(x_sample[i], y_sample[i])),
        nullptr,  // 核函数，这里不使用，为空
        abc       // 待估计参数
    );
  }

  // Solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;  // 输出到cout

  ceres::Solver::Summary summary;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);  // start optimization
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "total time used: " << time_used.count() << "sec" << endl;

  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a : abc) cout << a << " ";
  cout << endl;

  return 0;
}