#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
#ifdef DEBUG_
        std::cout << "kalman filter: prediction start." << std::endl;
#endif

    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
#ifdef DEBUG_
        std::cout << "kalman filter: prediction end." << std::endl;
#endif

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
#ifdef DEBUG_
        std::cout << "Update Start." << std::endl;
#endif

    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    UpdateCommon(y);

#ifdef DEBUG_
        std::cout << "Update End." << std::endl;
#endif
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * Handle non-linear scenario. 
    * update the state by using Extended Kalman Filter equations
  */
#ifdef DEBUG_
        std::cout << "UpdateEKF: Start." << std::endl;
#endif

    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);

    
    if(px < 0.001)
        px += 0.001;
    if(py < 0.001)
        py += 0.001;

    double rho = sqrt(px*px + py*py);
    double theta = atan2(py, px);
    double rho_der = (px*vx + py*vy)/rho;

    VectorXd z_pred = VectorXd(3);
    z_pred << rho, theta, rho_der;

    VectorXd y = z - z_pred;

    while(y(1) < -M_PI)
        y(1) += 2*M_PI;

    while(y(1) > M_PI)
        y(1) -= 2*M_PI;
    
    UpdateCommon(y); 
#ifdef DEBUG_
        std::cout << "UpdateEKF: End." << std::endl;
#endif

}

void KalmanFilter::UpdateCommon(const VectorXd &z){

    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //Calculate new estimate.
    x_ = x_ + (K * z);
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - K*H_) * P_;

}
