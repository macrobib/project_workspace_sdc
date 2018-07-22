#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse(estimations.size());
    rmse.setZero(estimations.size());

    if(estimations.size() != 0 &&(estimations.size() == ground_truth.size())){
        
        for(unsigned int i= 0; i < estimations.size(); ++i){
            
            VectorXd diff = estimations[i] - ground_truth[i];
            diff = diff.array() * diff.array();
            rmse += diff;
        }

        rmse = rmse/estimations.size();
        rmse = rmse.array().sqrt();
    }

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3, 4);
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    Hj.setZero(3, 4);
    double sqsum, sqm, sqcube, vel_vx_py, vel_vy_px;

    sqsum = px*px + py*py;
    if(!(fabs(px) > 0.001 && fabs(py) > 0.001)){
        sqm = 0.0001;
     }
    else{
        sqm = sqrt(sqsum);
    }
        sqcube = sqm * sqsum;
        vel_vx_py = vx*py - vy*px;
        vel_vy_px = vy*px - vx*py;

        Hj << (px/sqm), (py/sqm), 0, 0,
              (-1.0 * py/sqsum), (px/sqsum), 0, 0,
              (py * vel_vx_py/sqcube), (px * vel_vy_px/sqcube), px/sqm, py/sqm;

        return Hj;
}
