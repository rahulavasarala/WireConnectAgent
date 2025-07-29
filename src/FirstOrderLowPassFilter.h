#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class ButterworthLowPass {
public:
    ButterworthLowPass(double cutoffFrequency, double sampleRate, size_t dimension)
        : cutoffFrequency(cutoffFrequency), sampleRate(sampleRate), dimension(dimension) {
        calculateCoefficients();
        prevInput.setZero(dimension);
        prevOutput.setZero(dimension);
    }

    Eigen::VectorXd update(const Eigen::VectorXd& input) {
        // Ensure input has the correct dimensions
        if (input.size() != dimension) {
            throw std::invalid_argument("Input dimension does not match filter dimension.");
        }

        // Apply the filter equation
        Eigen::VectorXd output = a0 * input + a1 * prevInput - b1 * prevOutput;

        // Update previous values
        prevInput = input;
        prevOutput = output;

        return output;
    }

private:
    double cutoffFrequency;      // Cutoff frequency in Hz
    double sampleRate;           // Sampling rate in Hz
    size_t dimension;            // Dimension of the input vector
    double a0, a1, b1;           // Filter coefficients
    Eigen::VectorXd prevInput;   // Previous input vector
    Eigen::VectorXd prevOutput;  // Previous output vector

    void calculateCoefficients() {
        // Calculate filter coefficients for a first-order Butterworth low-pass filter
        double omega = 2.0 * M_PI * cutoffFrequency / sampleRate;
        double alpha = omega / (1.0 + omega);

        a0 = alpha;
        a1 = alpha;
        b1 = alpha - 1.0;
    }
};