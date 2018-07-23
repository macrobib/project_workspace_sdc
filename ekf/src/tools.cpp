#include <iostream>
#include "tools.h"
#define DEBUG_ 1

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
#ifdef DEBUG_
        std::cout << "CalculateRMSE: Start." << std::endl;
#endif
    
    long array_size = estimations.size();
    VectorXd rmse(4);
    rmse.setZero(4);
#ifdef DEBUG_
    std::cout << "Estimation array size: "<< array_size<< std::endl;
#endif

    if(array_size != 0 &&(array_size == ground_truth.size())){
        
        for(unsigned int i= 0; i < array_size; i++){
            
            VectorXd diff = estimations[i] - ground_truth[i];
            diff = diff.array() * diff.array();
            rmse += diff;
        }

        rmse = rmse/estimations.size();
        rmse = sqrt(rmse.array());
    }

    return rmse;
#ifdef DEBUG_
        std::cout << "CalculateRMSE: End" << std::endl;
#endif

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
#ifdef DEBUG_
        std::cout << "CalculateJacobian: Start." << std::endl;
#endif

    MatrixXd Hj(3, 4);
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double sqsum, sqm, sqcube, vel_vx_py, vel_vy_px;

    if(fabs(px) < 0.001 || fabs(py) < 0.001){
        px += 0.001;
        py += 0.001;
     }
    sqsum = px*px + py*py;
    sqm = sqrt(sqsum);
    sqcube = sqm * sqsum;
    vel_vx_py = vx*py - vy*px;
    vel_vy_px = vy*px - vx*py;

    Hj << (px/sqm), (py/sqm), 0, 0,
          (-py/sqsum), (px/sqsum), 0, 0,
          (py * vel_vx_py/sqcube), (px * vel_vy_px/sqcube), px/sqm, py/sqm;

    return Hj;
#ifdef DEBUG_
        std::cout << "CalculateJacobian: Stop." << std::endl;
#endif

}
