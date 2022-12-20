#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// Simulation parameters.
static const int kNumBodies = 16000;
static const float kMaxMass = 1000;
static const int kSeed = 42;
static const float kTimeInterval = 0.5;
static const int kBenchmarkIterations = 100;
static const float kDampeningFactor = 0.2f;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

#endif  // CONFIGURATION_H