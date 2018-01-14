#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  is_initialized_ = false;
  NIS_thr_ = 7.815;
  NIS_more_than_thr_ = 0;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  use_laser_ = true;  
  use_radar_ = true;  
  x_ = VectorXd(n_x_);  
  P_ = MatrixXd(n_x_, n_x_);
  std_a_ = 3;  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 0.5;  // Process noise standard deviation yaw acceleration in rad/s^2

  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  std_laspx_ = 0.15;  // Laser measurement noise standard deviation position1 in m
  std_laspy_ = 0.15;  // Laser measurement noise standard deviation position2 in m
  std_radr_ = 0.3;  // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03;  // Radar measurement noise standard deviation angle in rad
  std_radrd_ = 0.3;  // Radar measurement noise standard deviation radius change in m/s
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  auto s = [](double x){ return x*x; };
  Q_ = MatrixXd(2,2);
  Q_<<s(std_a_),0,0,s(std_yawdd_);
  R_laser_ = MatrixXd(2,2);
  R_laser_<<s(std_laspx_),0,0,s(std_laspy_);
  R_radar_ = MatrixXd(3,3);
  R_radar_<<s(std_radr_),0,0,0,s(std_radphi_),0,0,0,s(std_radrd_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(!is_initialized_) {
    stepid_ = 0;
    previous_timestamp_ = meas_package.timestamp_;
    x_ = VectorXd(n_x_);
    P_ = MatrixXd(n_x_,n_x_);
    if(meas_package.sensor_type_==MeasurementPackage::LASER) {
      double px(meas_package.raw_measurements_(0)),
	py(meas_package.raw_measurements_(1));
      x_<<px,py,0,0,0;
      P_ <<
	1,0,0,0,0,
	0,1,0,0,0,
	0,0,1000,0,0,
	0,0,0,M_PI,0,
	0,0,0,0,M_PI;
    } else if (meas_package.sensor_type_==MeasurementPackage::RADAR) {
      double ro(meas_package.raw_measurements_(0)),
	theta(meas_package.raw_measurements_(1)),
	ro_dot(meas_package.raw_measurements_(2));
      double ct(cos(theta)),st(sin(theta));
      x_ << ro*ct,ro*st,ro_dot/ct,0,0;
      P_ <<
	10,0,0,0,0,
	0,10,0,0,0,
	0,0,1,0,0,
	0,0,0,M_PI,0,
	0,0,0,0,M_PI;
    }
    is_initialized_ = true;
    return;
  }
  stepid_++;
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
  previous_timestamp_ = meas_package.timestamp_;  
}

VectorXd weights(int n,double lambda) {
  VectorXd w(2);
  double f=1.0/(lambda+n);
  w<<f*lambda,f*0.5;
  return w;
}

std::tuple<MatrixXd,VectorXd> sigmaPoints(const VectorXd& x,const MatrixXd& P,double lambda) {
  auto n = x.rows();
  assert(P.rows() == n && P.cols() == n);
  double m = sqrt(lambda+n);
  MatrixXd delta = P.llt().matrixL();
  delta*=m;
  MatrixXd Xsig(n,2*n+1);
  Xsig.col(0)  = x;
  for (int i = 0; i < n; i++) {
    Xsig.col(i+1)   = x + delta.col(i);
    Xsig.col(i+1+n) = x - delta.col(i);
  }
  return std::make_tuple(Xsig,weights(n,lambda));
}

VectorXd mean(const MatrixXd& X,const VectorXd& w) {
  VectorXd ret = X.col(0)*w(0);
  for(int i = 1;i<X.cols();i++)
    ret += X.col(i)*w(1);
  return ret;
}

MatrixXd covariance(const MatrixXd& X,const VectorXd& mu,const VectorXd& w) {
  MatrixXd ret = MatrixXd(X.rows(),X.rows());
  ret.fill(0.0);
  for(int i=0;i<X.cols();i++) {
    VectorXd delta = X.col(i)-mu;
    ret += delta*delta.transpose()*w(i==0?0:1);
  }
  return ret;
}

MatrixXd transform(const MatrixXd& X,const std::function<VectorXd(const VectorXd&)>& f) {
  auto n_in = X.rows();
  auto cols_in = X.cols();
  VectorXd fx = f(X.col(0));
  auto n_out = fx.rows();
  MatrixXd ret(n_out,cols_in);
  ret.col(0)=fx;
  for(int i = 1;i<cols_in;i++) 
    ret.col(i) = f(X.col(i));
  return ret;
}

MatrixXd crossCorrelation(const MatrixXd& X,const VectorXd& x,const MatrixXd& Z,const VectorXd& z,const VectorXd& w) {
  assert(x.cols()==z.cols());
  assert(x.rows()==X.rows());
  assert(z.rows()==Z.rows());
  MatrixXd ret(x.rows(),z.rows());
  ret.fill(0.0);
  for(int i=0;i<X.cols();i++) {
    VectorXd dx = X.col(i) - x;
    VectorXd dz = Z.col(i) - z;
    ret += dx*dz.transpose()*w(i==0?0:1);
  }
  return ret;
}

std::tuple<VectorXd,MatrixXd> transformBelief(const VectorXd& x,const MatrixXd& P,
					      const std::function<VectorXd(const VectorXd&)>& f,double lambda) {
  
  MatrixXd Xsig;
  VectorXd Xsig_w;
  std::tie(Xsig,Xsig_w) = sigmaPoints(x,P,lambda);
  MatrixXd fXsig = transform(Xsig,f);
  VectorXd mu = mean(fXsig,Xsig_w);
  MatrixXd cov = covariance(fXsig,mu,Xsig_w);
  return std::make_tuple(mu,cov);
}

VectorXd laserMeasurementModel(const VectorXd& x) {
  assert(x.rows()==5);
  VectorXd ret(2);
  ret<<x(0),x(1);
  return ret;
}

VectorXd radarMeasurementModel(const VectorXd& x) {
  assert(x.rows()==5);
  VectorXd ret(3);
  double px(x(0)),py(x(1)),v(x(2)),yaw(x(3)),yawd(x(4));
  double c1= px*px+py*py;
  double c2 = sqrt(c1);
  ret(0) = c2;
  ret(1) = c2>0?atan2(py,px):0.0/* should be improved */;
  ret(2) = c2>0.00001?v*(cos(yaw)*px+sin(yaw)*py)/c2:v;
  return ret;
}

VectorXd processModel(const VectorXd& x,double dt) {
  assert(x.rows()==7);
  VectorXd ret(5);
  double p_x = x(0);
  double p_y = x(1);
  double v = x(2);
  double yaw = x(3);
  double yawd = x(4);
  double nu_a = x(5);
  double nu_yawdd = x(6);
  
  //predicted state values
  double px_p, py_p;
  double v_p = v;
  double yaw_p = yaw + yawd*dt;
  double yawd_p = yawd;
  double sin_yaw = sin(yaw);
  double cos_yaw = cos(yaw);
  //avoid division by zero
  if (fabs(yawd) > 0.001) {
    double f = v/yawd;
    px_p = p_x + f * ( sin(yaw_p) - sin_yaw);
    py_p = p_y + f * ( cos_yaw - cos(yaw_p));
  }
  else {
    px_p = p_x + v*dt*cos_yaw;
    py_p = p_y + v*dt*sin_yaw;
  }
  //add noise
  {
    double f = 0.5*nu_a*dt*dt ;
    px_p = px_p + f * cos_yaw;
    py_p = py_p + f * sin_yaw;
  }
  v_p = v_p + nu_a*dt;
  yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
  yawd_p = yawd_p + nu_yawdd*dt;
  ret<< px_p,py_p,v_p,yaw_p,yawd_p;  
  return ret;
}

std::tuple<VectorXd,MatrixXd> augmentState(const VectorXd& x,const MatrixXd& P,const MatrixXd& Q) {
  assert(x.rows()==P.rows());
  assert(x.rows()==P.cols());
  assert(Q.rows()==Q.cols());
  int nx = x.rows();
  int d = Q.rows();
  int n_aug = nx+d;
  VectorXd x_aug(n_aug);
  MatrixXd P_aug(n_aug,n_aug);
  x_aug.head(nx) = x;
  x_aug.tail(d).fill(0);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(nx,nx) = P;
  P_aug.bottomRightCorner(d,d) = Q;
  return std::make_tuple(x_aug,P_aug);
}

std::tuple<VectorXd,MatrixXd,double> updateBeliefHelper(const MatrixXd& Xsig,const VectorXd& Xsig_w,const VectorXd& x,const MatrixXd& P,
							const VectorXd& zk,const MatrixXd& R,
							const std::function<VectorXd(const VectorXd&)>& measurementModel) {
  MatrixXd Zsig = transform(Xsig,measurementModel);
  VectorXd zk_p = mean(Zsig,Xsig_w);
  MatrixXd Sk_p = covariance(Zsig,zk_p,Xsig_w);
  VectorXd y = zk - zk_p;
  y(1) = fmod(y(1),2*M_PI);
  MatrixXd S = Sk_p + R;
  MatrixXd T = crossCorrelation(Xsig,x,Zsig,zk_p,Xsig_w);
  MatrixXd K = T*S.inverse();
  VectorXd xk = x + K*y;
  MatrixXd Pk = P - K*S*K.transpose();
  double NIS = y.transpose()*S.inverse()*y;
  return std::make_tuple(xk,Pk,NIS);
}

std::tuple<VectorXd,MatrixXd,double> predictAndUpdateBelief(const VectorXd& x,const MatrixXd& P,const MatrixXd& Q,
							    const VectorXd& zk,const MatrixXd& R, double lambda,
							    const std::function<VectorXd(const VectorXd&)>& processModel,
							    const std::function<VectorXd(const VectorXd&)>& measurementModel) {
  VectorXd x_aug;
  MatrixXd P_aug;
  std::tie(x_aug,P_aug) = augmentState(x,P,Q);
  MatrixXd Xsig;
  VectorXd Xsig_w;
  std::tie(Xsig,Xsig_w) = sigmaPoints(x_aug,P_aug,lambda);
  MatrixXd Xsig_p = transform(Xsig,processModel);
  VectorXd xk_p = mean(Xsig_p,Xsig_w);
  MatrixXd Pk_p = covariance(Xsig_p,xk_p,Xsig_w);
  return updateBeliefHelper(Xsig_p,Xsig_w,xk_p,Pk_p,zk,R,measurementModel);
}

std::tuple<VectorXd,MatrixXd,double> updateBelief(const VectorXd& x,const MatrixXd& P,
						  const VectorXd& zk,const MatrixXd& R,double lambda,
						  const std::function<VectorXd(const VectorXd&)>& measurementModel) {
  MatrixXd Xsig;
  VectorXd Xsig_w;
  std::tie(Xsig,Xsig_w) = sigmaPoints(x,P,lambda);
  return updateBeliefHelper(Xsig,Xsig_w,x,P,zk,R,measurementModel);
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  std::tie(x_,P_) = transformBelief(x_,P_,[delta_t](const VectorXd& x){return processModel(x,delta_t);},lambda_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  auto dt = 1e-6*(meas_package.timestamp_ - previous_timestamp_);
  double NIS;
  if(dt>0.0001) {
    if(!use_laser_) {
      Prediction(dt);
    } else {
      std::tie(x_,P_,NIS) = predictAndUpdateBelief(x_,P_,Q_,meas_package.raw_measurements_,R_laser_,lambda_,
						   [dt](const VectorXd& x){return processModel(x,dt);},
						   laserMeasurementModel);
    }
  } else if(use_laser_) {
    std::tie(x_,P_,NIS) = updateBelief(x_,P_,meas_package.raw_measurements_,R_laser_,lambda_,laserMeasurementModel);
  }
  if(NIS>NIS_thr_) NIS_more_than_thr_++;
  std::cout<<"L "<<stepid_<<" "<<NIS<<" "
	   <<NIS_more_than_thr_<<" "
	   <<NIS_more_than_thr_*100.0/(1.0+stepid_)<<endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  auto dt = 1e-6*(meas_package.timestamp_ - previous_timestamp_);
  double NIS;
  if(dt>0.0001) {
    if(!use_radar_) {
      Prediction(dt);
    } else {
      std::tie(x_,P_,NIS) = predictAndUpdateBelief(x_,P_,Q_,meas_package.raw_measurements_,R_radar_,lambda_,
						   [dt](const VectorXd& x){return processModel(x,dt);},
						   radarMeasurementModel);
    }
  } else if(use_radar_) {
    std::tie(x_,P_,NIS) = updateBelief(x_,P_,meas_package.raw_measurements_,R_radar_,lambda_,radarMeasurementModel);
  }
  if(NIS>NIS_thr_) NIS_more_than_thr_++;
  std::cout<<"R "<<stepid_<<" "<<NIS<<" "
	   <<NIS_more_than_thr_<<" "
	   <<NIS_more_than_thr_*100.0/(1.0+stepid_)<<endl;
}

