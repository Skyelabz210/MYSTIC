//!
//! INNOVATION: Quantum algorithms require NEGATIVE amplitudes for destructive
//! interference (Grover diffusion, quantum Fourier transform, etc.).
//!
//! Standard approach: Use unsigned integers, can't represent interference
//! Our approach: MobiusInt-backed amplitudes with explicit polarity
//!
//! This enables:
//! - Grover's algorithm diffusion operator (2|ψ⟩⟨ψ| - I)
//! - Quantum Fourier Transform (requires complex phases)
//! - Any algorithm requiring destructive interference

use crate::arithmetic::{MobiusInt, Polarity};

/// Quantum amplitude with signed magnitude via MobiusInt
#[derive(Clone, Copy, Debug)]
pub struct QuantumAmplitude {
    /// Signed value via MobiusInt (magnitude + polarity)
    pub value: MobiusInt,
}

impl QuantumAmplitude {
    /// Create zero amplitude
    #[inline]
    pub fn zero() -> Self {
        Self { value: MobiusInt::zero() }
    }
    
    /// Create positive amplitude
    #[inline]
    pub fn positive(magnitude: u64) -> Self {
        Self { value: MobiusInt::from_unsigned(magnitude, Polarity::Plus) }
    }
    
    /// Create negative amplitude (for destructive interference)
    #[inline]
    pub fn negative(magnitude: u64) -> Self {
        Self { value: MobiusInt::from_unsigned(magnitude, Polarity::Minus) }
    }
    
    /// Create from i64
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        Self { value: MobiusInt::from_i64(val) }
    }
    
    /// Get signed value
    #[inline]
    pub fn spinor_value(&self) -> i64 {
        self.value.spinor_value()
    }
    
    /// Get magnitude (absolute value)
    #[inline]
    pub fn magnitude(&self) -> u64 {
        self.value.abs()
    }
    
    /// Is positive?
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.value.is_positive()
    }
    
    /// Is negative?
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.value.is_negative()
    }
    
    /// Is zero?
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
    
    /// Flip sign (for oracle marking in Grover)
    #[inline]
    pub fn flip_sign(&self) -> Self {
        Self { value: self.value.neg() }
    }
    
    /// Add amplitudes (superposition)
    pub fn add(&self, other: &Self) -> Self {
        Self { value: self.value.add(&other.value) }
    }
    
    /// Subtract amplitudes
    pub fn sub(&self, other: &Self) -> Self {
        Self { value: self.value.sub(&other.value) }
    }
    
    /// Multiply amplitudes
    pub fn mul(&self, other: &Self) -> Self {
        Self { value: self.value.mul(&other.value) }
    }
    
    /// Scale by integer
    pub fn scale(&self, factor: i64) -> Self {