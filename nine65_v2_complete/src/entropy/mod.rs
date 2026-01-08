//! Entropy Module - QMNF Entropy System
//!
//! Two entropy sources for different purposes:
//!
//! ## Shadow Entropy (`shadow` module)
//! - Deterministic, reproducible randomness
//! - Use for: Testing, benchmarks, noise sampling
//! - Fast: <10ns per sample
//!
//! ## Secure Entropy (`secure` module) 
//! - OS CSPRNG (non-deterministic)
//! - Use for: Secret key generation, public key randomness
//! - Required for production security
//!
//! # Security Guidance (December 2024 Audit)
//!
//! | Operation | Use This |
//! |-----------|----------|
//! | Secret key generation | `secure::*` |
//! | Public key 'a' polynomial | `secure::*` |
//! | Error/noise sampling | `shadow::*` or `secure::*` |
//! | Testing/benchmarks | `shadow::*` |
//! | Reproducible results | `shadow::*` |

pub mod shadow;
pub mod secure;
pub mod wassan_noise;  // V2: 144 φ-harmonic holographic noise field

pub use shadow::ShadowHarvester;
pub use wassan_noise::WassanNoiseField;  // V2: O(1) noise retrieval
pub use secure::{
    secure_bytes, 
    secure_u64, 
    secure_u128,
    secure_u64_bounded, 
    secure_ternary,
    secure_cbd,
    secure_cbd_vector,
    secure_uniform_vector,
    secure_ternary_vector,
};
