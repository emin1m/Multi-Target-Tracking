# Multi-Target-Tracking
Multi Target Tracking Using JPDA
JPDA used as data association
For state estimation EKF is used.
# Multi-Target Tracking with EKF + JPDA

This repository contains a **multi-target radar tracking simulation**
based on a **Constant Turn (CT) motion model**, **Extended Kalman Filter (EKF)**,
and **Joint Probabilistic Data Association (JPDA)**.

The scenario focuses on tracking multiple maneuvering targets
under heavy clutter using nonlinear bearing–range measurements.

---

## Overview

A stationary radar observes multiple targets approaching the sensor
while executing coordinated turns.  
Each target is tracked independently, while data association is handled
jointly across all tracks.

The implementation is built using the **Stone Soup** tracking framework
and is structured in a modular, research-oriented way.

---

## State Representation

Each target is represented by a 5D state vector:

x = [ x, vx, y, vy, ω ]ᵀ

where:
- x, y   : Cartesian position
- vx, vy : velocity components
- ω      : turn rate

This state formulation allows modeling realistic curved trajectories.

---

## Motion Model

Target dynamics are modeled using a **Constant Turn (CT)** model with process noise:
- Linear acceleration noise on velocity components
- Turn-rate noise on ω

This enables smooth maneuvering motion while remaining compatible
with EKF-based filtering.

---

## Measurement Model

The radar provides nonlinear **bearing–range** measurements:

z = [ θ , r ]ᵀ

where:
- θ = atan2(y − ys, x − xs)
- r = sqrt((x − xs)² + (y − ys)²)

Measurement noise is modeled as zero-mean Gaussian in both bearing and range.

---

## Data Association (JPDA)

In cluttered environments, multiple measurements may be statistically
compatible with multiple tracks.

**JPDA** evaluates all feasible track–measurement association hypotheses
and computes their probabilities based on:
- Measurement likelihoods
- Detection probability
- Clutter spatial density

Each track is updated using a probability-weighted combination of
hypotheses, followed by Gaussian mixture reduction to maintain
a single Gaussian state.

---

## Track Management

- Tracks are initialized with uncertain state estimates
- Track confirmation is based on consecutive detection hits
- Missed detections are counted and tracks are deleted afte
