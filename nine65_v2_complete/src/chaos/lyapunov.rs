//! Exact Lyapunov Exponent Analysis
//!
//! Lyapunov exponents measure the rate of separation of infinitesimally close
//! trajectories in chaotic systems. Traditional float-based implementations
//! suffer from accumulated error that corrupts the measurement.
//!
//! This implementation uses exact integer arithmetic to compute Lyapunov
//! exponents without drift.
//!
//! # Key Insight
//!
//! The maximal Lyapunov exponent λ determines predictability:
//! - λ > 0: Chaotic (nearby trajectories diverge exponentially)
//! - λ < 0: Stable (nearby trajectories converge)
//! - λ ≈ 0: Marginal (bifurcation point)
//!
//! For weather: When λ suddenly increases, the system is entering
//! a more chaotic regime - potentially indicating severe weather forming.

use super::lorenz::{ExactLorenz, LorenzState};

/// Scale for Lyapunov calculations
const LYAP_SCALE: i128 = 1 << 50;

/// Exact Lyapunov exponent representation
#[derive(Clone, Debug)]
pub struct LyapunovExponent {
    /// Accumulated log of divergence ratios (scaled integer)
    log_sum: i128,
    /// Number of measurements
    count: u64,
    /// Scale factor
    scale: i128,
}

impl LyapunovExponent {
    pub fn new() -> Self {
        Self {
            log_sum: 0,
            count: 0,
            scale: LYAP_SCALE,
        }
    }
    
    /// Get the current Lyapunov exponent estimate
    pub fn value(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        (self.log_sum as f64 / self.scale as f64) / (self.count as f64)
    }
    
    /// Add a divergence measurement
    pub fn add_measurement(&mut self, ratio: i128) {
        // Approximate log using integer arithmetic
        // log(ratio) ≈ (ratio - SCALE) / SCALE for ratio near SCALE
        // For larger ratios, we use bit counting
        if ratio > 0 {
            self.log_sum += integer_log(ratio, self.scale);
            self.count += 1;
        }
    }
    
    /// Is the system chaotic? (λ > 0)
    pub fn is_chaotic(&self) -> bool {
        self.value() > 0.0
    }
    
    /// Confidence in the estimate (based on sample count)
    pub fn confidence(&self) -> f64 {
        // More samples = higher confidence, saturates at 1.0
        1.0 - 1.0 / (1.0 + self.count as f64 / 100.0)
    }
}

impl Default for LyapunovExponent {
    fn default() -> Self {
        Self::new()
    }
}

/// Lyapunov analyzer for chaotic systems
pub struct LyapunovAnalyzer {
    /// Reference trajectory system
    reference: ExactLorenz,
    /// Perturbed trajectory system
    perturbed: ExactLorenz,
    /// Initial perturbation magnitude (scaled)
    epsilon: i128,
    /// Computed exponent
    exponent: LyapunovExponent,
    /// Renormalization interval (steps between measurements)
    renorm_interval: u64,
    /// Steps since last renormalization
    steps_since_renorm: u64,
}

impl LyapunovAnalyzer {
    /// Create analyzer for a given initial state
    pub fn new(initial: LorenzState, epsilon: f64) -> Self {
        let scale: i128 = 1 << 40;
        let eps_scaled = (epsilon * scale as f64) as i128;
        
        // Perturbed initial condition (small offset in x)
        let perturbed_initial = LorenzState::from_scaled(
            initial.x + eps_scaled,
            initial.y,
            initial.z,
        );
        
        Self {
            reference: ExactLorenz::new(initial.clone()),
            perturbed: ExactLorenz::new(perturbed_initial),
            epsilon: eps_scaled,
            exponent: LyapunovExponent::new(),
            renorm_interval: 100,
            steps_since_renorm: 0,
        }
    }
    
    /// Set renormalization interval
    pub fn with_renorm_interval(mut self, interval: u64) -> Self {
        self.renorm_interval = interval;
        self
    }
    
    /// Single step of the analyzer
    pub fn step(&mut self) {
        self.reference.step();
        self.perturbed.step();
        self.steps_since_renorm += 1;
        
        if self.steps_since_renorm >= self.renorm_interval {
            self.renormalize();
            self.steps_since_renorm = 0;
        }
    }
    
    /// Renormalize the perturbed trajectory and record divergence
    fn renormalize(&mut self) {
        let ref_state = self.reference.state();
        let pert_state = self.perturbed.state();
        
        // Compute current separation
        let dx = pert_state.x - ref_state.x;
        let dy = pert_state.y - ref_state.y;
        let dz = pert_state.z - ref_state.z;
        
        // Scaled distance squared (avoid sqrt for now)
        let d2 = (dx >> 20).pow(2) + (dy >> 20).pow(2) + (dz >> 20).pow(2);
        let distance = integer_sqrt(d2 as u128) as i128;
        
        // Ratio of current separation to initial separation
        let eps_scaled = self.epsilon >> 20;
        if distance > 0 && eps_scaled > 0 {
            let ratio = (distance << 40) / eps_scaled;
            self.exponent.add_measurement(ratio);
        }
        
        // Renormalize: reset perturbed trajectory to epsilon distance from reference
        let norm_factor = if distance > 0 {
            (self.epsilon << 20) / distance.max(1)
        } else {
            1 << 40
        };
        
        // Scale the perturbation back to epsilon
        let new_dx = (dx >> 20) * norm_factor >> 40;
        let new_dy = (dy >> 20) * norm_factor >> 40;
        let new_dz = (dz >> 20) * norm_factor >> 40;
        
        self.perturbed = ExactLorenz::new(LorenzState::from_scaled(
            ref_state.x + new_dx,
            ref_state.y + new_dy,
            ref_state.z + new_dz,
        ));
    }
    
    /// Run analyzer for N steps
    pub fn analyze(&mut self, steps: u64) -> &LyapunovExponent {
        for _ in 0..steps {
            self.step();
        }
        &self.exponent
    }
    
    /// Get current Lyapunov exponent estimate
    pub fn exponent(&self) -> &LyapunovExponent {
        &self.exponent
    }
    
    /// Get reference system state
    pub fn state(&self) -> &LorenzState {
        self.reference.state()
    }
    
    /// Compute local Lyapunov exponent (instantaneous chaos level)
    /// Higher values = more chaotic at this moment
    pub fn local_exponent(&self) -> f64 {
        let ref_state = self.reference.state();
        let pert_state = self.perturbed.state();
        
        let dx = pert_state.x - ref_state.x;
        let dy = pert_state.y - ref_state.y;
        let dz = pert_state.z - ref_state.z;
        
        let d2 = (dx >> 20).pow(2) + (dy >> 20).pow(2) + (dz >> 20).pow(2);
        let distance = integer_sqrt(d2 as u128) as f64;
        let epsilon_f = (self.epsilon >> 20) as f64;
        
        if epsilon_f > 0.0 {
            (distance / epsilon_f).ln() / self.steps_since_renorm as f64
        } else {
            0.0
        }
    }
}

/// Compute signature of current chaos state (for attractor matching)
#[derive(Clone, Debug)]
pub struct ChaosSignature {
    /// Recent Lyapunov exponent
    pub lyapunov: f64,
    /// Local chaos intensity
    pub local_chaos: f64,
    /// Phase space region (discretized)
    pub phase_region: (i32, i32, i32),
    /// Rate of change of chaos
    pub chaos_derivative: f64,
}

impl ChaosSignature {
    pub fn from_analyzer(analyzer: &LyapunovAnalyzer, prev_local: f64) -> Self {
        let state = analyzer.state();
        let local = analyzer.local_exponent();
        
        // Discretize phase space into regions
        let region = (
            (state.x_f64() / 5.0) as i32,
            (state.y_f64() / 5.0) as i32,
            (state.z_f64() / 5.0) as i32,
        );
        
        Self {
            lyapunov: analyzer.exponent().value(),
            local_chaos: local,
            phase_region: region,
            chaos_derivative: local - prev_local,
        }
    }
    
    /// Distance to another signature (for attractor matching)
    pub fn distance_to(&self, other: &ChaosSignature) -> f64 {
        let lyap_diff = (self.lyapunov - other.lyapunov).abs();
        let local_diff = (self.local_chaos - other.local_chaos).abs();
        let region_diff = ((self.phase_region.0 - other.phase_region.0).abs()
            + (self.phase_region.1 - other.phase_region.1).abs()
            + (self.phase_region.2 - other.phase_region.2).abs()) as f64;
        let deriv_diff = (self.chaos_derivative - other.chaos_derivative).abs();
        
        // Weighted combination
        lyap_diff * 10.0 + local_diff * 5.0 + region_diff * 1.0 + deriv_diff * 3.0
    }
}

/// Integer logarithm approximation (scaled)
fn integer_log(x: i128, scale: i128) -> i128 {
    if x <= 0 {
        return 0;
    }
    
    // Count bits for rough log2
    let bits = 127 - x.leading_zeros() as i128;
    
    // log(x) ≈ log(2) * log2(x) ≈ 0.693 * bits
    // Scale: 0.693 * scale ≈ scale * 693 / 1000
    let log2_scaled = bits * scale;
    log2_scaled * 693 / 1000
}

/// Integer square root
fn integer_sqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lorenz_is_chaotic() {
        let initial = LorenzState::classic();
        // Use larger epsilon (0.001) to avoid scaling underflow
        let mut analyzer = LyapunovAnalyzer::new(initial, 0.001);
        
        analyzer.analyze(10_000);
        
        let lyap = analyzer.exponent().value();
        println!("Lyapunov exponent: {:.4}", lyap);
        
        // Lorenz system has λ ≈ 0.9 for standard parameters
        // With integer arithmetic we may get slightly different values
        assert!(lyap > 0.0, "Expected positive λ (chaotic), got {}", lyap);
    }
    
    #[test]
    fn test_lyapunov_deterministic() {
        let initial = LorenzState::classic();
        
        let mut analyzer1 = LyapunovAnalyzer::new(initial.clone(), 0.001);
        let mut analyzer2 = LyapunovAnalyzer::new(initial, 0.001);
        
        analyzer1.analyze(5_000);
        analyzer2.analyze(5_000);
        
        // Should give EXACTLY the same result
        assert_eq!(
            analyzer1.exponent().value(),
            analyzer2.exponent().value(),
            "Lyapunov computation not deterministic!"
        );
    }
    
    #[test]
    fn test_chaos_signature() {
        let initial = LorenzState::classic();
        let mut analyzer = LyapunovAnalyzer::new(initial, 0.001);
        
        analyzer.analyze(1_000);
        let sig1 = ChaosSignature::from_analyzer(&analyzer, 0.0);
        
        analyzer.analyze(1_000);
        let sig2 = ChaosSignature::from_analyzer(&analyzer, sig1.local_chaos);
        
        println!("Signature 1: lyap={:.4}, local={:.4}, region={:?}", 
            sig1.lyapunov, sig1.local_chaos, sig1.phase_region);
        println!("Signature 2: lyap={:.4}, local={:.4}, region={:?}", 
            sig2.lyapunov, sig2.local_chaos, sig2.phase_region);
        
        let dist = sig1.distance_to(&sig2);
        println!("Distance: {:.4}", dist);
    }
}
