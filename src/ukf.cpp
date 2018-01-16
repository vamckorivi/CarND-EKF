#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cstdlib>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI / 4;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:
    Complete the initialization. See ukf.h for other member properties.*/ // init state dimensions

    n_x_ = 5;
    lambda_ = 3 - n_x_;
    n_aug_ = 7;

    // init state vector
    x_ = VectorXd(5);

    x_ << 0.0,
            0.0,
            0.0,
            0.0,
            0.0;

    // init covariance matrix P_
    P_ = MatrixXd(n_x_, n_x_);

    // Sigma points
    n_sigma_ = 2 * n_aug_ + 1;
    // init weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    // init Xsig_pred_ =
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // time initializaion
    time_us_ = 0.0;

    // init Radar covariance matrix
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

    // init Laser covariance matrix
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    // init NIS for Radar
    NIS_radar_ = 0.0;

    // init NIS for Laser
    NIS_lidar_ = 0.0;




}

UKF::~UKF() {}

void UKF::Angle(double *angle) {
    while (*angle > M_PI) * angle -= 2. *M_PI;
    while (*angle < -M_PI) * angle += 2. *M_PI;
}
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {


    if (!is_initialized_) {
        // covariance matrix independent of sensor
        P_ = MatrixXd::Identity(5, 5);

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rhodot = meas_package.raw_measurements_[2];
            // polar to cartesion coodinate conversion
            double px = rho * cos(phi);
            double py = rho * sin(phi);
            double v = sqrt((rhodot*rhodot)*(sin(phi)*sin(phi) + cos(phi)*cos(phi)));
            x_ << px, py, v, 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // avoid zeroes
            if (fabs(x_(0)) < 0.001 and fabs(x_(1)) < 0.001) { x_(0) = x_(1) = 0.001; }
        }

        // init weights
        weights_(0) = lambda_ / (lambda_ + n_aug_);
        for (int i = 1; i < n_sigma_; i++) { weights_(i) = 0.5 / (lambda_ + n_aug_); }

        // init timestep
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        std::cout << "Initialization completed" << endl;
    }

    double dt = (meas_package.timestamp_ - time_us_)/1000000.0;

    //start the timer for next dt
    time_us_ = meas_package.timestamp_;

    //predict
    Prediction(dt);

    //update measurements for radar and lidar

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    }
}

void UKF::Prediction(double delta_t) {
    /**
    TODO:
    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */


    double delta_tsq = delta_t*delta_t;
    // augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    // augmented covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    // augmented sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    x_aug.fill(0.0);
    x_aug.head(n_x_) = x_;

    P_aug.fill(0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;

    // building Q matrix
    MatrixXd Q = MatrixXd(2, 2);
    Q << std_a_*std_a_, 0,
            0, std_yawdd_*std_yawdd_;
    P_aug.bottomRightCorner(2, 2) = Q;

    // create square root matrix
    MatrixXd P_sq = P_aug.llt().matrixL();
    Xsig_aug.col(0) = x_aug;

    double sqrt_lambda_n_aug = sqrt(lambda_ + n_aug_);

    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i + 1) = x_aug + sqrt_lambda_n_aug * P_sq.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt_lambda_n_aug* P_sq.col(i);
    }


    for (int i = 0; i < n_sigma_; i++)
    {

        const double px = Xsig_aug(0, i);
        const double py = Xsig_aug(1, i);
        const double v = Xsig_aug(2, i);
        const double psi = Xsig_aug(3, i);
        const double psidot = Xsig_aug(4, i);
        const double mu_a = Xsig_aug(5, i);
        const double mu_psi = Xsig_aug(6, i);

        double px_p, py_p;
        if (fabs(psidot) > 0.001) {
            double v_psidot = v / psidot;
            px_p = px + v_psidot*(sin(psi + psidot*delta_t) - sin(psi));
            py_p = py + v_psidot*(cos(psi) - cos(psi + psidot*delta_t));
        }

        else {
            double v_dt = v*delta_t;
            px_p = px + v_dt*cos(psi);
            py_p = py + v_dt*sin(psi);
        }
        double v_p = v;
        double psi_p = psi + psidot*delta_t;
        double psidot_p = psidot;
        // noise calculation
        px_p += 0.5*mu_a*delta_tsq*cos(psi);
        py_p += 0.5*mu_a*delta_tsq*sin(psi);
        v_p += mu_a*delta_t;
        psi_p += 0.5*mu_psi*delta_tsq;
        psidot_p += 0.5*mu_psi*delta_t;

        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = psi_p;
        Xsig_pred_(4, i) = psidot_p;

    }


    x_ = Xsig_pred_ * weights_;
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        //calculating diff between pred and measurement
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //normalize angeles
        Angle(&(x_diff(3)));
        P_ += weights_(i)*x_diff*x_diff.transpose();
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:
    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    You'll also need to calculate the lidar NIS.
    */

    int n_dim = 2;
    MatrixXd Zsig = Xsig_pred_.block(0, 0, n_dim, n_sigma_);
    UpdateUKF(meas_package, Zsig, n_dim);
}





void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:
    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    You'll also need to calculate the radar NIS.
    */

    int n_dim = 3;
    MatrixXd Zsig = MatrixXd(n_dim, n_sigma_);
    // convert and calculate sigma pt values for Rader
    for (int i = 0; i < n_sigma_; i++) {
        const double px = Xsig_pred_(0, i);
        const double py = Xsig_pred_(1, i);
        const double v = Xsig_pred_(2, i);
        const double psi = Xsig_pred_(3, i);
        // Filling measurements for rho phi and rhodot
        Zsig(0, i) = sqrt(px*px + py*py);
        Zsig(1, i) = atan2(py, py);
        Zsig(2, i) = (px*cos(psi)*v + py*sin(psi)*v) / sqrt(px*px + py*py);
    }
    UpdateUKF(meas_package, Zsig, n_dim);
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_dim) {

    // calculating measurement means
    VectorXd z_pred = VectorXd(n_dim);
    z_pred = Zsig * weights_;
    // measurement covariance matrix
    MatrixXd S = MatrixXd(n_dim, n_dim);
    S.fill(0.0);
    // building residuals
    for (int i = 0; i<n_sigma_; i++) {
        // calculating diff between measurement prediction and sigma pts
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // normalize angles
        Angle(&(z_diff(1)));
        S += weights_(i)*z_diff*z_diff.transpose();
    }

    // Noise covariance calculations

    MatrixXd R = MatrixXd(n_dim, n_dim);
    // switch for laser and radar:
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { R = R_radar_; }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) { R = R_laser_; }

    // Combining noise with measurement covariance
    S += R;

    // computing cross correlation Tc for Kalman Boost
    MatrixXd Tc = MatrixXd(n_x_, n_dim);
    Tc.fill(0.0);
    for (int i = 0; i<n_sigma_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization for radar phi
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { Angle(&(z_diff(1))); }

        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        Angle(&(x_diff(3)));
        Tc += weights_(i)*x_diff*z_diff.transpose();
    }
    //Kalman Gain
    VectorXd z = meas_package.raw_measurements_;
    MatrixXd K = Tc * S.inverse();

    VectorXd z_diff = z - z_pred;
    // angle normalization for radar phi
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { Angle(&(z_diff(1))); }

    // Updating State Mean Vector and State Covariance Matrix
    x_ += K*z_diff;
    P_ -= K*S*K.transpose();

    // NIS Measurements
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        NIS_radar_ = z.transpose() * S.inverse() * z;
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        NIS_lidar_ = z.transpose() * S.inverse() * z;
    }

}