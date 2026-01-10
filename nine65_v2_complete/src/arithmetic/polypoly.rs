//! PolyPoly - Optimized polynomial multiplication strategies.
//!
//! Provides schoolbook, Karatsuba, and NTT-based multiplication for
//! integer-only polynomials modulo q.

#[cfg(feature = "ntt_fft")]
use super::ntt_fft::NTTEngineFFT as NTTEngine;

#[cfg(not(feature = "ntt_fft"))]
use super::ntt::NTTEngine;

use std::ops::{Add, Sub};

/// Polynomial in Z_q[X]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial {
    pub coeffs: Vec<i64>,
    pub modulus: u64,
}

impl Polynomial {
    /// Create polynomial from coefficients
    pub fn from_coeffs(coeffs: Vec<i64>, modulus: u64) -> Self {
        Polynomial { coeffs, modulus }
    }

    /// Create zero polynomial of degree n
    pub fn zero(degree: usize, modulus: u64) -> Self {
        Polynomial {
            coeffs: vec![0; degree],
            modulus,
        }
    }

    /// Degree of polynomial (highest non-zero coefficient)
    pub fn degree(&self) -> Option<usize> {
        for i in (0..self.coeffs.len()).rev() {
            if self.coeffs[i] != 0 {
                return Some(i);
            }
        }
        None
    }

    /// Reduce all coefficients mod q
    pub fn reduce(&mut self) {
        let q = self.modulus as i64;
        for coeff in &mut self.coeffs {
            *coeff = coeff.rem_euclid(q);
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        let q = self.modulus as i64;
        let mut result = self.clone();
        for coeff in &mut result.coeffs {
            *coeff = (*coeff * scalar).rem_euclid(q);
        }
        result
    }

    /// Evaluate polynomial at point x
    pub fn evaluate(&self, x: i64) -> i64 {
        let q = self.modulus as i64;
        let mut result = 0i64;
        for &coeff in self.coeffs.iter().rev() {
            result = (result * x + coeff).rem_euclid(q);
        }
        result
    }

    /// Negate polynomial (for subtraction)
    pub fn negate(&self) -> Self {
        let q = self.modulus as i64;
        let mut result = self.clone();
        for coeff in &mut result.coeffs {
            *coeff = (-*coeff).rem_euclid(q);
        }
        result
    }
}

/// Polynomial addition
impl Add for Polynomial {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus);
        let q = self.modulus as i64;
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).unwrap_or(&0);
            let b = other.coeffs.get(i).unwrap_or(&0);
            let sum = (a + b).rem_euclid(q);
            result.push(sum);
        }

        Polynomial {
            coeffs: result,
            modulus: self.modulus,
        }
    }
}

/// Polynomial subtraction
impl Sub for Polynomial {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other.negate()
    }
}

/// Polynomial multiplication strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiplicationStrategy {
    Schoolbook,
    Karatsuba,
    NTT,
    Auto,
}

/// PolyPoly: Optimized polynomial multiplication
pub struct PolyPolyMultiplier {
    modulus: u64,
}

impl PolyPolyMultiplier {
    pub fn new(modulus: u64) -> Self {
        PolyPolyMultiplier { modulus }
    }

    /// Multiply two polynomials (auto-select strategy)
    pub fn multiply(&self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        self.multiply_with_strategy(a, b, MultiplicationStrategy::Auto)
    }

    /// Multiply with specific strategy
    pub fn multiply_with_strategy(
        &self,
        a: &Polynomial,
        b: &Polynomial,
        strategy: MultiplicationStrategy,
    ) -> Polynomial {
        assert_eq!(a.modulus, self.modulus);
        assert_eq!(b.modulus, self.modulus);

        match strategy {
            MultiplicationStrategy::Schoolbook => self.schoolbook_multiply(a, b),
            MultiplicationStrategy::Karatsuba => self.karatsuba_multiply(a, b),
            MultiplicationStrategy::NTT => self.ntt_multiply(a, b),
            MultiplicationStrategy::Auto => {
                let degree = a.coeffs.len().max(b.coeffs.len());
                if degree < 64 {
                    self.schoolbook_multiply(a, b)
                } else if degree < 512 {
                    self.karatsuba_multiply(a, b)
                } else {
                    self.ntt_multiply(a, b)
                }
            }
        }
    }

    /// Schoolbook multiplication: O(n²)
    fn schoolbook_multiply(&self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        if a.coeffs.is_empty() || b.coeffs.is_empty() {
            return Polynomial::zero(0, self.modulus);
        }
        let q = self.modulus as i64;
        let result_len = a.coeffs.len() + b.coeffs.len() - 1;
        let mut result = vec![0i64; result_len];

        for (i, &a_coeff) in a.coeffs.iter().enumerate() {
            for (j, &b_coeff) in b.coeffs.iter().enumerate() {
                let prod = (a_coeff * b_coeff).rem_euclid(q);
                result[i + j] = (result[i + j] + prod).rem_euclid(q);
            }
        }

        Polynomial {
            coeffs: result,
            modulus: self.modulus,
        }
    }

    /// Karatsuba multiplication: O(n^1.585)
    fn karatsuba_multiply(&self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        if a.coeffs.len() <= 32 || b.coeffs.len() <= 32 {
            return self.schoolbook_multiply(a, b);
        }

        let n = a.coeffs.len().max(b.coeffs.len());
        let m = (n + 1) / 2;

        let (a0, a1) = self.split_at(a, m);
        let (b0, b1) = self.split_at(b, m);

        let z0 = self.karatsuba_multiply(&a0, &b0);
        let z2 = self.karatsuba_multiply(&a1, &b1);

        let a_sum = a0.clone() + a1.clone();
        let b_sum = b0.clone() + b1.clone();
        let z1_full = self.karatsuba_multiply(&a_sum, &b_sum);

        let z1 = z1_full - z0.clone() - z2.clone();

        let mut result = z0.clone();

        for (i, &coeff) in z1.coeffs.iter().enumerate() {
            let idx = i + m;
            while result.coeffs.len() <= idx {
                result.coeffs.push(0);
            }
            result.coeffs[idx] =
                (result.coeffs[idx] + coeff).rem_euclid(self.modulus as i64);
        }

        for (i, &coeff) in z2.coeffs.iter().enumerate() {
            let idx = i + 2 * m;
            while result.coeffs.len() <= idx {
                result.coeffs.push(0);
            }
            result.coeffs[idx] =
                (result.coeffs[idx] + coeff).rem_euclid(self.modulus as i64);
        }

        result
    }

    /// NTT-based multiplication: O(n log n)
    fn ntt_multiply(&self, a: &Polynomial, b: &Polynomial) -> Polynomial {
        if a.coeffs.is_empty() || b.coeffs.is_empty() {
            return Polynomial::zero(0, self.modulus);
        }

        let result_len = a.coeffs.len() + b.coeffs.len() - 1;
        let n = result_len.next_power_of_two();
        if !self.ntt_compatible(n) {
            return self.karatsuba_multiply(a, b);
        }

        let q = self.modulus as i64;
        let mut a_padded = vec![0u64; n];
        let mut b_padded = vec![0u64; n];

        for (i, &coeff) in a.coeffs.iter().enumerate() {
            a_padded[i] = coeff.rem_euclid(q) as u64;
        }
        for (i, &coeff) in b.coeffs.iter().enumerate() {
            b_padded[i] = coeff.rem_euclid(q) as u64;
        }

        let ntt = NTTEngine::new(self.modulus, n);
        let product = ntt.multiply(&a_padded, &b_padded);
        let coeffs: Vec<i64> = product[..result_len].iter().map(|&c| c as i64).collect();

        Polynomial {
            coeffs,
            modulus: self.modulus,
        }
    }

    /// Polynomial division (returns quotient, remainder)
    pub fn divide(&self, dividend: &Polynomial, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        let mut quotient = Polynomial::zero(dividend.coeffs.len(), self.modulus);
        let mut remainder = dividend.clone();

        while remainder.degree().is_some()
            && divisor.degree().is_some()
            && remainder.degree().unwrap() >= divisor.degree().unwrap()
        {
            let r_deg = remainder.degree().unwrap();
            let d_deg = divisor.degree().unwrap();
            let deg_diff = r_deg - d_deg;

            let leading_coeff = remainder.coeffs[r_deg];
            let divisor_leading = divisor.coeffs[d_deg];
            let divisor_leading_inv = self.mod_inverse(divisor_leading as u64, self.modulus);

            let coeff = (leading_coeff * divisor_leading_inv as i64)
                .rem_euclid(self.modulus as i64);
            if deg_diff < quotient.coeffs.len() {
                quotient.coeffs[deg_diff] = coeff;
            }

            for i in 0..=d_deg {
                let sub = (divisor.coeffs[i] * coeff).rem_euclid(self.modulus as i64);
                remainder.coeffs[i + deg_diff] =
                    (remainder.coeffs[i + deg_diff] - sub).rem_euclid(self.modulus as i64);
            }

            remainder.coeffs[r_deg] = 0;
        }

        (quotient, remainder)
    }

    fn split_at(&self, poly: &Polynomial, m: usize) -> (Polynomial, Polynomial) {
        let low = poly.coeffs[..m.min(poly.coeffs.len())].to_vec();
        let high = if poly.coeffs.len() > m {
            poly.coeffs[m..].to_vec()
        } else {
            vec![0]
        };

        (
            Polynomial::from_coeffs(low, self.modulus),
            Polynomial::from_coeffs(high, self.modulus),
        )
    }

    fn ntt_compatible(&self, n: usize) -> bool {
        if n == 0 {
            return false;
        }
        (self.modulus - 1) % (2 * n as u64) == 0
    }

    fn mod_inverse(&self, a: u64, m: u64) -> u64 {
        let (mut old_r, mut r) = (a as i64, m as i64);
        let (mut old_s, mut s) = (1i64, 0i64);

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;
        }

        ((old_s % m as i64 + m as i64) % m as i64) as u64
    }
}

/// PolyPoly convolution (optimized for FHE)
pub struct PolyPolyConvolution {
    multiplier: PolyPolyMultiplier,
}

impl PolyPolyConvolution {
    pub fn new(modulus: u64) -> Self {
        PolyPolyConvolution {
            multiplier: PolyPolyMultiplier::new(modulus),
        }
    }

    /// Cyclic convolution (for X^N + 1 reduction)
    pub fn cyclic_convolution(&self, a: &Polynomial, b: &Polynomial, n: usize) -> Polynomial {
        let product = self.multiplier.multiply(a, b);
        self.reduce_mod_xn_plus_1(product, n)
    }

    /// Negacyclic convolution (for FHE)
    pub fn negacyclic_convolution(&self, a: &Polynomial, b: &Polynomial, n: usize) -> Polynomial {
        self.cyclic_convolution(a, b, n)
    }

    fn reduce_mod_xn_plus_1(&self, poly: Polynomial, n: usize) -> Polynomial {
        let mut result = vec![0i64; n];
        let q = poly.modulus as i64;

        for (i, &coeff) in poly.coeffs.iter().enumerate() {
            let idx = i % n;
            let quotient = i / n;

            if quotient % 2 == 0 {
                result[idx] = (result[idx] + coeff).rem_euclid(q);
            } else {
                result[idx] = (result[idx] - coeff).rem_euclid(q);
            }
        }

        Polynomial {
            coeffs: result,
            modulus: poly.modulus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_addition() {
        let a = Polynomial::from_coeffs(vec![1, 2, 3], 7);
        let b = Polynomial::from_coeffs(vec![4, 5], 7);

        let sum = a + b;
        assert_eq!(sum.coeffs, vec![5, 0, 3]);
    }

    #[test]
    fn test_polynomial_subtraction() {
        let a = Polynomial::from_coeffs(vec![5, 2, 3], 7);
        let b = Polynomial::from_coeffs(vec![1, 2, 1], 7);

        let diff = a - b;
        assert_eq!(diff.coeffs, vec![4, 0, 2]);
    }

    #[test]
    fn test_schoolbook_multiply() {
        let multiplier = PolyPolyMultiplier::new(7);
        let a = Polynomial::from_coeffs(vec![1, 2], 7);
        let b = Polynomial::from_coeffs(vec![3, 4], 7);

        let product = multiplier.schoolbook_multiply(&a, &b);
        assert_eq!(product.coeffs, vec![3, 3, 1]);
    }

    #[test]
    fn test_karatsuba_multiply() {
        let multiplier = PolyPolyMultiplier::new(11);
        let a = Polynomial::from_coeffs(vec![1, 2, 3, 4], 11);
        let b = Polynomial::from_coeffs(vec![5, 6, 7], 11);

        let product1 = multiplier.schoolbook_multiply(&a, &b);
        let product2 = multiplier.karatsuba_multiply(&a, &b);

        assert_eq!(product1.coeffs, product2.coeffs);
    }

    #[test]
    fn test_auto_strategy() {
        let multiplier = PolyPolyMultiplier::new(17);

        let a_small = Polynomial::from_coeffs(vec![1; 32], 17);
        let b_small = Polynomial::from_coeffs(vec![2; 32], 17);
        let _ = multiplier.multiply(&a_small, &b_small);

        let a_large = Polynomial::from_coeffs(vec![1; 256], 17);
        let b_large = Polynomial::from_coeffs(vec![2; 256], 17);
        let _ = multiplier.multiply(&a_large, &b_large);
    }

    #[test]
    fn test_polynomial_evaluation() {
        let poly = Polynomial::from_coeffs(vec![1, 2, 3], 11);
        assert_eq!(poly.evaluate(2), 6);
    }

    #[test]
    fn test_cyclic_convolution() {
        let conv = PolyPolyConvolution::new(7);
        let a = Polynomial::from_coeffs(vec![1, 2, 3, 4], 7);
        let b = Polynomial::from_coeffs(vec![5, 6, 7, 8], 7);

        let result = conv.cyclic_convolution(&a, &b, 4);
        assert_eq!(result.coeffs.len(), 4);
    }

    #[test]
    fn test_polynomial_division() {
        let multiplier = PolyPolyMultiplier::new(11);

        let dividend = Polynomial::from_coeffs(vec![1, 2, 3, 4], 11);
        let divisor = Polynomial::from_coeffs(vec![1, 1], 11);

        let (quotient, remainder) = multiplier.divide(&dividend, &divisor);
        let reconstructed = multiplier.multiply(&quotient, &divisor) + remainder;

        for (i, &coeff) in dividend.coeffs.iter().enumerate() {
            if i < reconstructed.coeffs.len() {
                assert_eq!(coeff, reconstructed.coeffs[i]);
            }
        }
    }
}
