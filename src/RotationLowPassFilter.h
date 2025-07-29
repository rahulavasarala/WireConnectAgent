#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

class RotationLowPassFilter {
private:
    Eigen::Quaterniond filteredQuat; // Current filtered quaternion
    double alpha;                    // Low-pass filter coefficient (0 < alpha < 1)
    bool initialized;                // Indicates if the filter is initialized

    // Calculate alpha from cutoff frequency and sampling rate
    double computeAlpha(double cutoffFreq, double samplingRate) {
        if (cutoffFreq <= 0 || samplingRate <= 0) {
            throw std::invalid_argument("Cutoff frequency and sampling rate must be positive.");
        }
        double rc = 1.0 / (2.0 * M_PI * cutoffFreq); // Time constant (RC)
        double dt = 1.0 / samplingRate;             // Sampling period
        return dt / (dt + rc);                      // Derived alpha
    }

public:
    // Constructor using cutoff frequency and sampling rate
    RotationLowPassFilter(double cutoffFreq, double samplingRate)
        : alpha(computeAlpha(cutoffFreq, samplingRate)), initialized(false) {}

    // Reset the filter with an initial rotation
    void reset(const Eigen::Matrix3d& initialRotation) {
        filteredQuat = Eigen::Quaterniond(initialRotation);
        filteredQuat.normalize();
        initialized = true;
    }

    // Apply the filter to a new rotation
    Eigen::Matrix3d filter(const Eigen::Matrix3d& newRotation) {
        if (!initialized) {
            reset(newRotation); // Initialize with the first rotation
        }

        // Convert the new rotation to a quaternion
        Eigen::Quaterniond newQuat(newRotation);
        newQuat.normalize();

        // Apply SLERP for low-pass filtering
        filteredQuat = filteredQuat.slerp(alpha, newQuat);
        filteredQuat.normalize();

        // Return the filtered rotation as a matrix
        return filteredQuat.toRotationMatrix();
    }
};