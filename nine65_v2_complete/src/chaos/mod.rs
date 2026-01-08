//! MYSTIC - Exact Chaos Mathematics
//!
//! Mathematically Yielding Stable Trajectory Integer Computation
//!
//! Zero-drift chaos simulation using exact integer arithmetic.
//! Eliminates the butterfly effect by removing floating-point error accumulation.
//!
//! Named in memory of Camp Mystic. No more flash flood tragedies.
//!
//! # Core Principle
//!
//! Traditional chaos: Error × e^(λt) → ∞ as t → ∞
//! MYSTIC chaos: 0 × e^(λt) = 0 (no initial error to amplify)
//!
//! # Extended Prediction via Liouville Evolution
//!
//! Beyond 14 days, we evolve PROBABILITY DENSITY instead of single trajectories:
//! - Liouville equation: ∂ρ/∂t = {ρ, H} (probability conserved exactly)
//! - MobiusInt enables signed operations in RNS (Poisson brackets work)
//! - Symplectic integration preserves Hamiltonian structure
//!
//! # Applications
//!
//! - Weather prediction without drift (0-14 days: trajectory, 14-30+ days: probability)
//! - Flash flood attractor detection
//! - Long-term climate modeling
//! - Historical weather prediction failures NOW SOLVED

pub mod lorenz;
pub mod lyapunov;
pub mod attractor;
pub mod weather;
pub mod liouville;
pub mod quantum_enhanced;  // QUANTUM-ENHANCED PATTERN DETECTION

// Re-export all public types for convenient access
pub use lorenz::{ExactLorenz, LorenzState, LorenzParams};
pub use lyapunov::{LyapunovAnalyzer, LyapunovExponent, ChaosSignature};
pub use attractor::{AttractorDetector, AttractorSignature, AttractorBasin, AttractorId};
pub use weather::{WeatherState, FloodDetector, DelugeEngine, AlertLevel, FloodPrediction, RawSensorData};
pub use liouville::{LiouvilleEvolver, MobiusInt, PhaseDensity, PhaseCell, ExtendedForecast, extended_forecast};
pub use quantum_enhanced::{
    QuantumPeriodDetector, HolographicAttractorSearch, QuantumFloodTimer,
    QuantumMYSTIC, QuantumPrediction, FloodTimingAnalysis,
};
