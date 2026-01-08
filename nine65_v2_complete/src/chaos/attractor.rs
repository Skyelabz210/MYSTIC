//! Attractor Detection and Basin Analysis
//!
//! This module identifies when a chaotic system enters specific attractor basins.
//! For weather prediction, this means detecting when atmospheric conditions
//! enter a "flash flood attractor" - the set of conditions that historically
//! produce severe flooding.
//!
//! # Key Concept
//!
//! Instead of predicting exact rainfall amounts (impossible due to chaos),
//! we detect when the system enters a BASIN of attraction that historically
//! leads to floods. This works because:
//!
//! 1. Chaotic systems have structure (strange attractors)
//! 2. Trajectories within a basin evolve similarly
//! 3. Basin entry can be detected BEFORE the event manifests
//!
//! This gives hours of warning instead of minutes.

use super::lyapunov::ChaosSignature;
use super::lorenz::LorenzState;
use std::collections::HashMap;
use std::fs;

use serde::Deserialize;

/// Unique identifier for an attractor
pub type AttractorId = u64;

/// Signature of a known attractor (learned from historical data)
#[derive(Clone, Debug)]
pub struct AttractorSignature {
    /// Unique identifier
    pub id: AttractorId,
    /// Human-readable name
    pub name: String,
    /// Characteristic Lyapunov exponent range
    pub lyapunov_min: f64,
    pub lyapunov_max: f64,
    /// Characteristic phase space regions
    pub regions: Vec<(i32, i32, i32)>,
    /// Chaos derivative signature (increasing = approaching critical point)
    pub typical_derivative: f64,
    /// Severity level (0-10)
    pub severity: u8,
    /// Historical occurrences used to build this signature
    pub sample_count: u64,
}

impl AttractorSignature {
    /// Create a new attractor signature
    pub fn new(id: AttractorId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            lyapunov_min: 0.0,
            lyapunov_max: f64::MAX,
            regions: Vec::new(),
            typical_derivative: 0.0,
            severity: 0,
            sample_count: 0,
        }
    }
    
    /// Add a chaos signature observation to this attractor
    pub fn add_observation(&mut self, sig: &ChaosSignature) {
        // Update Lyapunov bounds
        if sig.lyapunov < self.lyapunov_min || self.sample_count == 0 {
            self.lyapunov_min = sig.lyapunov;
        }
        if sig.lyapunov > self.lyapunov_max || self.sample_count == 0 {
            self.lyapunov_max = sig.lyapunov;
        }
        
        // Add region if not already present
        if !self.regions.contains(&sig.phase_region) {
            self.regions.push(sig.phase_region);
        }
        
        // Rolling average of derivative
        let n = self.sample_count as f64;
        self.typical_derivative = (self.typical_derivative * n + sig.chaos_derivative) / (n + 1.0);
        
        self.sample_count += 1;
    }
    
    /// Match score: how well does a signature match this attractor?
    /// Returns 0.0-1.0 (higher = better match)
    pub fn match_score(&self, sig: &ChaosSignature) -> f64 {
        let mut score = 0.0;
        let mut weights = 0.0;
        
        // Lyapunov in range?
        if sig.lyapunov >= self.lyapunov_min && sig.lyapunov <= self.lyapunov_max {
            score += 3.0;
        } else {
            let dist = if sig.lyapunov < self.lyapunov_min {
                self.lyapunov_min - sig.lyapunov
            } else {
                sig.lyapunov - self.lyapunov_max
            };
            score += 3.0 * (-dist).exp();
        }
        weights += 3.0;
        
        // Region match?
        let region_match = self.regions.iter()
            .map(|r| {
                let dx = (r.0 - sig.phase_region.0).abs();
                let dy = (r.1 - sig.phase_region.1).abs();
                let dz = (r.2 - sig.phase_region.2).abs();
                (-(dx + dy + dz) as f64 / 3.0).exp()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        score += 2.0 * region_match;
        weights += 2.0;
        
        // Derivative match?
        let deriv_diff = (sig.chaos_derivative - self.typical_derivative).abs();
        score += 1.0 * (-deriv_diff).exp();
        weights += 1.0;
        
        score / weights
    }
}

/// Basin of attraction - geometric region in phase space
#[derive(Clone, Debug)]
pub struct AttractorBasin {
    /// Center point (scaled integers)
    pub center: (i128, i128, i128),
    /// Radius (scaled)
    pub radius: i128,
    /// Associated attractor
    pub attractor_id: AttractorId,
    /// Severity level (0-10)
    pub severity: u8,
}

#[derive(Deserialize)]
struct BasinEntry {
    center: [f64; 3],
    radii: [f64; 3],
    #[allow(dead_code)]
    sample_count: Option<u64>,
}

fn alert_to_severity(alert: &str) -> u8 {
    match alert.to_uppercase().as_str() {
        "CLEAR" => 1,
        "WATCH" => 3,
        "ADVISORY" => 5,
        "WARNING" => 7,
        "EMERGENCY" => 9,
        _ => 5,
    }
}

impl AttractorBasin {
    /// Check if a state is within this basin
    pub fn contains(&self, state: &LorenzState) -> bool {
        let dx = state.x - self.center.0;
        let dy = state.y - self.center.1;
        let dz = state.z - self.center.2;
        
        // Squared distance (avoid sqrt)
        let d2 = (dx >> 20).pow(2) + (dy >> 20).pow(2) + (dz >> 20).pow(2);
        let r2 = (self.radius >> 20).pow(2);
        
        d2 <= r2
    }
    
    /// Distance to basin boundary (negative = inside)
    pub fn distance_to_boundary(&self, state: &LorenzState) -> f64 {
        let dx = state.x - self.center.0;
        let dy = state.y - self.center.1;
        let dz = state.z - self.center.2;
        
        let d2 = (dx >> 20).pow(2) + (dy >> 20).pow(2) + (dz >> 20).pow(2);
        let d = (d2 as f64).sqrt();
        let r = (self.radius >> 20) as f64;
        
        d - r
    }
}

/// Attractor detection engine
pub struct AttractorDetector {
    /// Known attractor signatures
    signatures: HashMap<AttractorId, AttractorSignature>,
    /// Known basins
    basins: Vec<AttractorBasin>,
    /// Detection threshold (0.0-1.0)
    threshold: f64,
    /// Next attractor ID
    next_id: AttractorId,
}

impl AttractorDetector {
    pub fn new() -> Self {
        Self {
            signatures: HashMap::new(),
            basins: Vec::new(),
            threshold: 0.7,
            next_id: 1,
        }
    }
    
    /// Set detection threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
    
    /// Register a known attractor
    pub fn register_attractor(&mut self, name: &str, severity: u8) -> AttractorId {
        let id = self.next_id;
        self.next_id += 1;
        
        let mut sig = AttractorSignature::new(id, name);
        sig.severity = severity;
        self.signatures.insert(id, sig);
        
        id
    }
    
    /// Add observation to an attractor
    pub fn add_observation(&mut self, attractor_id: AttractorId, sig: &ChaosSignature) {
        if let Some(attractor) = self.signatures.get_mut(&attractor_id) {
            attractor.add_observation(sig);
        }
    }
    
    /// Register a basin for an attractor
    pub fn register_basin(&mut self, attractor_id: AttractorId, center: (f64, f64, f64), radius: f64, severity: u8) {
        let scale: i128 = 1 << 40;
        let basin = AttractorBasin {
            center: (
                (center.0 * scale as f64) as i128,
                (center.1 * scale as f64) as i128,
                (center.2 * scale as f64) as i128,
            ),
            radius: (radius * scale as f64) as i128,
            attractor_id,
            severity,
        };
        self.basins.push(basin);
    }
    
    /// Detect which attractor (if any) the current state is approaching
    pub fn detect(&self, sig: &ChaosSignature) -> Option<(AttractorId, f64)> {
        let mut best_match: Option<(AttractorId, f64)> = None;
        
        for (id, attractor) in &self.signatures {
            let score = attractor.match_score(sig);
            if score >= self.threshold {
                if best_match.is_none() || score > best_match.as_ref().unwrap().1 {
                    best_match = Some((*id, score));
                }
            }
        }
        
        best_match
    }
    
    /// Check if state is in any known basin
    pub fn in_basin(&self, state: &LorenzState) -> Option<&AttractorBasin> {
        self.basins.iter().find(|b| b.contains(state))
    }
    
    /// Get distance to nearest dangerous basin
    pub fn distance_to_danger(&self, state: &LorenzState, min_severity: u8) -> Option<f64> {
        self.basins.iter()
            .filter(|b| b.severity >= min_severity)
            .map(|b| b.distance_to_boundary(state))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Load refined basin boundaries from JSON produced by training scripts.
    pub fn load_basins_from_file(&mut self, attractor_id: AttractorId, path: &str) -> Result<usize, String> {
        let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let basins: HashMap<String, BasinEntry> = serde_json::from_str(&contents).map_err(|e| e.to_string())?;

        let mut loaded = 0usize;
        for (alert, entry) in basins {
            let severity = alert_to_severity(&alert);
            let radius = entry.radii.iter().cloned().fold(0.0, f64::max);
            self.register_basin(
                attractor_id,
                (entry.center[0], entry.center[1], entry.center[2]),
                radius,
                severity,
            );
            loaded += 1;
        }

        Ok(loaded)
    }
    
    /// Get attractor by ID
    pub fn get_attractor(&self, id: AttractorId) -> Option<&AttractorSignature> {
        self.signatures.get(&id)
    }
    
    /// List all registered attractors
    pub fn list_attractors(&self) -> Vec<&AttractorSignature> {
        self.signatures.values().collect()
    }
}

impl Default for AttractorDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-built detector with common weather attractors
pub fn weather_detector() -> AttractorDetector {
    let mut detector = AttractorDetector::new();
    
    // Flash flood attractor (high severity)
    let flood_id = detector.register_attractor("FlashFlood", 9);
    
    // Severe thunderstorm attractor
    let storm_id = detector.register_attractor("SevereThunderstorm", 7);
    
    // Stable fair weather attractor
    let fair_id = detector.register_attractor("FairWeather", 1);
    
    // Note: Actual basins and signatures would be learned from historical data
    // These are placeholder values for the framework
    
    detector
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attractor_registration() {
        let mut detector = AttractorDetector::new();
        
        let id1 = detector.register_attractor("TestAttractor1", 5);
        let id2 = detector.register_attractor("TestAttractor2", 8);
        
        assert_ne!(id1, id2);
        assert_eq!(detector.get_attractor(id1).unwrap().name, "TestAttractor1");
        assert_eq!(detector.get_attractor(id2).unwrap().severity, 8);
    }
    
    #[test]
    fn test_basin_containment() {
        let scale: i128 = 1 << 40;
        let basin = AttractorBasin {
            center: (10 * scale, 10 * scale, 25 * scale),
            radius: 5 * scale,
            attractor_id: 1,
            severity: 5,
        };
        
        // Inside basin
        let inside = LorenzState::from_scaled(12 * scale, 11 * scale, 26 * scale);
        assert!(basin.contains(&inside));
        
        // Outside basin
        let outside = LorenzState::from_scaled(20 * scale, 10 * scale, 25 * scale);
        assert!(!basin.contains(&outside));
    }
    
    #[test]
    fn test_signature_matching() {
        let mut sig = AttractorSignature::new(1, "Test");
        
        // Add some observations
        let obs1 = ChaosSignature {
            lyapunov: 0.9,
            local_chaos: 0.8,
            phase_region: (2, 2, 5),
            chaos_derivative: 0.01,
        };
        sig.add_observation(&obs1);
        
        let obs2 = ChaosSignature {
            lyapunov: 1.0,
            local_chaos: 0.85,
            phase_region: (2, 3, 5),
            chaos_derivative: 0.02,
        };
        sig.add_observation(&obs2);
        
        // Good match
        let test_good = ChaosSignature {
            lyapunov: 0.95,
            local_chaos: 0.82,
            phase_region: (2, 2, 5),
            chaos_derivative: 0.015,
        };
        let score_good = sig.match_score(&test_good);
        
        // Poor match
        let test_poor = ChaosSignature {
            lyapunov: 2.5,
            local_chaos: 0.1,
            phase_region: (-5, -5, 0),
            chaos_derivative: -0.5,
        };
        let score_poor = sig.match_score(&test_poor);
        
        println!("Good match score: {:.3}", score_good);
        println!("Poor match score: {:.3}", score_poor);
        
        assert!(score_good > score_poor);
        assert!(score_good > 0.5);
    }
}
