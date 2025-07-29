/**
 * 
 */

#include "ForceSpaceParticleFilter.h"

#include <iostream>

using namespace Eigen;
using namespace std;
using namespace WireConnectAgent;

ForceSpaceParticleFilter::ForceSpaceParticleFilter(const int n_particles)
{
	_n_particles = n_particles;
	for(int i = 0; i < _n_particles; i++) {
		_particles.push_back(Vector3d::Zero());
		_particles_with_weight.push_back(make_pair(Vector3d::Zero(),1));
	}

	_mean_scatter = 0.0;
	_std_scatter = 0.025;

	_coeff_friction = 0.0;

	_F_low = 0.0;
	_F_high = 3.0;
	_v_low = 0.001;
	_v_high = 0.01;

	_F_low_add = 3.0;
	_F_high_add = 10.0;
	_v_low_add = 0.001;
	_v_high_add = 0.005;

	_alpha_add = 0.50;
	_alpha_remove = 0.10;
}

void ForceSpaceParticleFilter::reset() {
	_particles.resize(0);
	_particles_with_weight.resize(0);
	for (int i = 0; i < _n_particles; i++) {
		_particles.push_back(Vector3d::Zero());
		_particles_with_weight.push_back(make_pair(Vector3d::Zero(),1));
	}
}

void ForceSpaceParticleFilter::update(const Vector3d motion_control, const Vector3d force_control,
			const Vector3d velocity_measured, const Vector3d force_measured)
{

	resamplingLowVariance(motionUpdateAndWeighting(motion_control, force_control, velocity_measured, force_measured));
	computePCA();

	if(_eigenvalues.norm() < sqrt(0.5 * _n_particles) )
	{
		_force_space_dimension = 0;
	}
	else
	{

		double force_space_dimension_hb = 1;
		double force_space_dimension_lb = 3;
		
		for(int i=0 ; i<2 ; i++)
		{
			if(_eigenvalues(i) > _alpha_add * _eigenvalues(2))
			{
				force_space_dimension_hb++;
			}
			if(_eigenvalues(i) < _alpha_remove * _eigenvalues(2))
			{
				force_space_dimension_lb--;
			}
		}
		
		if(_force_space_dimension >= force_space_dimension_lb)
		{
			_force_space_dimension = force_space_dimension_lb;
		}
		else if(_force_space_dimension <= force_space_dimension_hb)
		{
			_force_space_dimension = force_space_dimension_hb;
		}
	}
}

Matrix3d ForceSpaceParticleFilter::getSigmaForce()
{

	if(_force_space_dimension == 0)
	{
		return Matrix3d::Zero();
	}
	else if(_force_space_dimension == 1)
	{
		Vector3d force_axis = _eigenvectors.col(2);
		return force_axis * force_axis.transpose();
	}
	else if(_force_space_dimension == 2)
	{
		Vector3d motion_axis = _eigenvectors.col(0);
		return Matrix3d::Identity() - motion_axis * motion_axis.transpose();
	}
	else
	{
		return Matrix3d::Identity();
	}

}

std::vector<Vector3d> ForceSpaceParticleFilter::getForceAxes() {
	std::vector<Vector3d> force_axes;

	if(_force_space_dimension == 0)
	{
		return force_axes;
	}
	else if(_force_space_dimension == 1)
	{
		force_axes.push_back(_eigenvectors.col(2));
	}
	else if(_force_space_dimension == 2)
	{
		force_axes.push_back(_eigenvectors.col(1));
		force_axes.push_back(_eigenvectors.col(2));
		// Vector3d motion_axis = _eigenvectors.col(0);
	}
	else
	{
		// return Matrix3d::Identity();
		force_axes.push_back(_eigenvectors.col(0));
		force_axes.push_back(_eigenvectors.col(1));
		force_axes.push_back(_eigenvectors.col(2));
	}
	return force_axes;
}

Vector3d ForceSpaceParticleFilter::getForceOrMotionAxis() {
	if (_force_space_dimension == 0) {
		return Vector3d::Zero();
	} else if (_force_space_dimension == 1) {
		return _eigenvectors.col(2);
	} else if (_force_space_dimension == 2) {
		return _eigenvectors.col(0);
	} else {
		return Vector3d::Zero();
	}
}

vector<pair<Vector3d, double>> ForceSpaceParticleFilter::motionUpdateAndWeighting(const Vector3d motion_control, const Vector3d force_control,
			const Vector3d velocity_measured, const Vector3d force_measured)
{
	Vector3d motion_control_normalized = Vector3d::Zero();
	Vector3d force_control_normalized = Vector3d::Zero();
	Vector3d measured_velocity_normalized = Vector3d::Zero();
	Vector3d measured_force_normalized = Vector3d::Zero();

	if(motion_control.norm() > 0.001)
	{
		motion_control_normalized = motion_control/motion_control.norm();
	}
	if(force_control.norm() > 0.001)
	{
		force_control_normalized = force_control/force_control.norm();
	}
	if(velocity_measured.norm() > 1e-3)
	{
		measured_velocity_normalized = velocity_measured/velocity_measured.norm();
	}
	if(force_measured.norm() > 0.5)
	{
		measured_force_normalized = force_measured/force_measured.norm();
	}

	vector<Vector3d> augmented_particles = _particles;

	// add particles at the center in case of contact loss
	int n_added_particles_center = _n_particles * 0.01;
	for(int i=0 ; i<n_added_particles_center ; i++)
	{
		augmented_particles.push_back(Vector3d::Zero());
	}

	// add particles in the direction of the motion control if there is no velocity in that direction
	double prob_add_particle = 0;
	if(motion_control.norm() - _coeff_friction * force_control.norm() > 0)  // take friction into account
	{
		prob_add_particle = wf(motion_control_normalized, force_measured, _F_low_add, _F_high_add) * wv(motion_control_normalized, velocity_measured, _v_low_add, _v_high_add);
	}
	if(prob_add_particle < 0)
	{
		prob_add_particle = 0;
	}

	int n_added_particles = prob_add_particle * _n_particles;
	for(int i=0 ; i<n_added_particles ; i++)
	{
		double alpha = (double) (i + 0.5) / (double)n_added_particles; // add particles on the arc betwen the motion and force control
		Vector3d new_particle = (1 - alpha) * motion_control_normalized + alpha * force_control_normalized;
		new_particle.normalize();
		augmented_particles.push_back(new_particle);
	}


	int n_new_particles = n_added_particles_center + n_added_particles;

	// prepare weights
	vector<pair<Vector3d, double>> augmented_weighted_particles;

	for(int i=0 ; i< _n_particles + n_new_particles ; i++)
	{
		// control update : scatter the particles that are not at the center
		Vector3d current_particle = augmented_particles[i];

		if(current_particle.norm() > 1e-3) // contact
		{
			double normal_rand_1 = sampleNormalDistribution(_mean_scatter, _std_scatter);
			double normal_rand_2 = sampleNormalDistribution(_mean_scatter, _std_scatter);
			double normal_rand_3 = sampleNormalDistribution(_mean_scatter, _std_scatter);
			current_particle += Vector3d(normal_rand_1, normal_rand_2, normal_rand_3);

			current_particle.normalize();
		}

		// measurement update : compute weight due to force measurement
		double weight_force = wf(current_particle, force_measured, _F_low, _F_high);

		// measurement update : compute weight due to velocity measurement
		double weight_velocity = wv(current_particle, velocity_measured, _v_low, _v_high);

		// final weight
		double weight = weight_force * weight_velocity;

		augmented_weighted_particles.push_back(make_pair(current_particle, weight));
	}

	return augmented_weighted_particles;
}

void ForceSpaceParticleFilter::resamplingLowVariance(vector<pair<Vector3d, double>> augmented_weighted_particles)
{
	int n_augmented_weighted_particles = augmented_weighted_particles.size();
	vector<double> cumulative_weights;

	double sum_of_weights = 0;
	for(int i=0 ; i<n_augmented_weighted_particles ; i++)
	{
		sum_of_weights += augmented_weighted_particles[i].second;
		cumulative_weights.push_back(sum_of_weights);
	}

	for(int i=0 ; i<n_augmented_weighted_particles ; i++)
	{
		cumulative_weights[i] /= sum_of_weights;
	}

	double n_inv = 1.0/(double)_n_particles;
	double r = sampleUniformDistribution(0,n_inv);
	int k = 0;

	for(int i=0 ; i<_n_particles ; i++)
	{
		while(r > cumulative_weights[k])
		{
			k++;
		}
		_particles[i] = augmented_weighted_particles[k].first;

		_particles_with_weight[i].first = augmented_weighted_particles[k].first;
		_particles_with_weight[i].second = augmented_weighted_particles[k].second;

		r += n_inv;
	}
}

void ForceSpaceParticleFilter::computePCA()
{
	MatrixXd points_to_PCA = MatrixXd::Zero(3, 1.5*_n_particles);
	for(int i=0 ; i<_n_particles ; i++)
	{
		points_to_PCA.col(i) = _particles[i];
	}

	MatrixXd centered_points_to_PCA = points_to_PCA.colwise() - points_to_PCA.rowwise().mean();
	Matrix3d cov = centered_points_to_PCA * centered_points_to_PCA.transpose();

	SelfAdjointEigenSolver<MatrixXd> eig(cov);

	// Get the eigenvectors and eigenvalues.
	_eigenvectors = eig.eigenvectors();
	_eigenvalues = eig.eigenvalues();
}

double ForceSpaceParticleFilter::sampleNormalDistribution(const double mean, const double std)
{
	// random device class instance, source of 'true' randomness for initializing random seed
    random_device rd; 
    // Mersenne twister PRNG, initialized with seed from random device instance
    mt19937 gen(rd()); 
    // instance of class normal_distribution with specific mean and stddev
    normal_distribution<float> d(mean, std); 
    // get random number with normal distribution using gen as random source
    return d(gen); 
}

double ForceSpaceParticleFilter::sampleUniformDistribution(const double min, const double max)
{
	double min_internal = min;
	double max_internal = max;
	if(min > max)
	{
		min_internal = max;
		max_internal = min;
	}
	// random device class instance, source of 'true' randomness for initializing random seed
    random_device rd; 
    // Mersenne twister PRNG, initialized with seed from random device instance
    mt19937 gen(rd()); 
    // instance of class uniform_distribution with specific min and max
    uniform_real_distribution<float> d(min_internal, max_internal); 
    // get random number with normal distribution using gen as random source
    return d(gen); 
}


double ForceSpaceParticleFilter::wf(const Vector3d particle, const Vector3d force_measured, const double fl, const double fh)
{
	double wf = 0;

	if(particle.norm() < 0.1)
	{
		wf = 1.0 - (force_measured.norm() - fl) / (fh - fl);
	}
	else
	{
		wf = (particle.dot(force_measured) - fl) / (fh - fl);
	}

	if(wf > 1) {wf = 1;}
	if(wf < 0) {wf = 0;}

	return wf;

}

double ForceSpaceParticleFilter::wv(const Vector3d particle, const Vector3d velocity_measured, const double vl, const double vh)
{
	double wv = 0.5;
	if(particle.norm() > 0.001)
	{
		wv = 1 - (particle.dot(velocity_measured) - vl) / (vh - vl);
	}

	if(wv > 1) {wv = 1;}
	if(wv < 0) {wv = 0;}

	return wv;
}