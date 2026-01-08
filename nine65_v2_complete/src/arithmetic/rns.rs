//! RNS - Residue Number System with Adaptive Multi-Prime Support
//!
//! QMNF Innovation: CRTBigInt parallel coefficient operations enable
//! exact arithmetic on integers larger than any single prime modulus.

use super::montgomery::MontgomeryContext;
#[cfg(feature = "ntt_fft")]
use super::ntt_fft::NTTEngineFFT as NTTEngine;

#[cfg(not(feature = "ntt_fft"))]
use super::ntt::NTTEngine;

/// RNS Context for managing multiple prime moduli
pub struct RNSContext {
    /// List of coprime moduli
    pub primes: Vec<u64>,
    /// Product of all primes (as u128, may overflow for many primes)
    pub product: u128,
    /// Montgomery contexts for each prime
    pub mont_contexts: Vec<MontgomeryContext>,
    /// NTT engines for each prime
    pub ntt_engines: Vec<NTTEngine>,
    /// Polynomial degree
    pub n: usize,
    /// Precomputed CRT reconstruction values
    /// For each prime q_i: M_i = (product / q_i), M_i_inv = M_i^(-1) mod q_i
    pub crt_values: Vec<(u128, u64)>,
}

impl RNSContext {
    /// Create a new RNS context from a list of primes
    pub fn new(primes: Vec<u64>, n: usize) -> Self {
        assert!(!primes.is_empty(), "Need at least one prime");
        assert!(n.is_power_of_two(), "N must be power of 2");
        
        // Verify primes are coprime (all distinct primes are coprime)
        for i in 0..primes.len() {
            for j in (i + 1)..primes.len() {
                assert_ne!(primes[i], primes[j], "Primes must be distinct");
            }
        }
        
        // Compute product
        let product = primes.iter().fold(1u128, |acc, &p| acc * p as u128);
        
        // Create Montgomery and NTT contexts
        let mont_contexts: Vec<_> = primes.iter()
            .map(|&p| MontgomeryContext::new(p))
            .collect();
        
        let ntt_engines: Vec<_> = primes.iter()
            .map(|&p| NTTEngine::new(p, n))
            .collect();
        
        // Precompute CRT values
        let crt_values: Vec<_> = primes.iter()
            .map(|&qi| {
                let mi = product / qi as u128;
                let mi_mod_qi = (mi % qi as u128) as u64;
                let mi_inv = mod_inverse(mi_mod_qi, qi);
                (mi, mi_inv)
            })
            .collect();
        
        Self {
            primes,
            product,
            mont_contexts,
            ntt_engines,
            n,
            crt_values,
        }
    }
    
    /// Get the number of primes
    pub fn num_primes(&self) -> usize {
        self.primes.len()
    }
    
    /// Convert a small integer to RNS representation
    pub fn from_int(&self, x: u64) -> Vec<u64> {
        self.primes.iter().map(|&q| x % q).collect()
    }
    
    /// Convert RNS representation back to integer using CRT
    /// Only valid if result < product of primes
    pub fn to_int(&self, rns: &[u64]) -> u128 {
        assert_eq!(rns.len(), self.primes.len());
        
        let mut result = 0u128;
        for i in 0..self.primes.len() {
            let (mi, mi_inv) = self.crt_values[i];
            let term = (rns[i] as u128 * mi_inv as u128) % self.primes[i] as u128;
            let contribution = (term * mi) % self.product;
            result = (result + contribution) % self.product;
        }
        
        result
    }
    
    /// Add two RNS numbers
    pub fn add(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.primes.len());
        assert_eq!(b.len(), self.primes.len());
        
        a.iter().zip(b.iter()).zip(self.primes.iter())
            .map(|((&ai, &bi), &qi)| {
                let sum = ai as u128 + bi as u128;
                if sum >= qi as u128 { (sum - qi as u128) as u64 } else { sum as u64 }
            })
            .collect()
    }
    
    /// Subtract two RNS numbers
    pub fn sub(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.primes.len());
        assert_eq!(b.len(), self.primes.len());
        
        a.iter().zip(b.iter()).zip(self.primes.iter())
            .map(|((&ai, &bi), &qi)| {
                if ai >= bi { ai - bi } else { qi - bi + ai }
            })
            .collect()
    }
    
    /// Multiply two RNS numbers
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.primes.len());
        assert_eq!(b.len(), self.primes.len());
        
        a.iter().zip(b.iter()).zip(self.primes.iter())
            .map(|((&ai, &bi), &qi)| {
                ((ai as u128 * bi as u128) % qi as u128) as u64
            })
            .collect()
    }
    
    /// Negate an RNS number
    pub fn neg(&self, a: &[u64]) -> Vec<u64> {
        a.iter().zip(self.primes.iter())
            .map(|(&ai, &qi)| {
                if ai == 0 { 0 } else { qi - ai }
            })
            .collect()
    }
}

/// RNS Polynomial - coefficients stored as parallel limbs
pub struct RNSPolynomial {
    /// Limbs: limbs[i] is the polynomial mod primes[i]
    pub limbs: Vec<Vec<u64>>,
    /// Polynomial degree
    pub n: usize,
}

impl RNSPolynomial {
    /// Create from a single-modulus polynomial
    pub fn from_poly(poly: &[u64], ctx: &RNSContext) -> Self {
        let n = poly.len();
        let limbs = ctx.primes.iter()
            .map(|&q| poly.iter().map(|&c| c % q).collect())
            .collect();
        
        Self { limbs, n }
    }
    
    /// Create a zero polynomial
    pub fn zero(ctx: &RNSContext) -> Self {
        let limbs = vec![vec![0u64; ctx.n]; ctx.num_primes()];
        Self { limbs, n: ctx.n }
    }
    
    /// Add two RNS polynomials
    pub fn add(&self, other: &Self, ctx: &RNSContext) -> Self {
        assert_eq!(self.n, other.n);
        
        let limbs = self.limbs.iter()
            .zip(other.limbs.iter())
            .zip(ctx.primes.iter())
            .map(|((a, b), &q)| {
                a.iter().zip(b.iter())
                    .map(|(&ai, &bi)| {
                        let sum = ai as u128 + bi as u128;
                        if sum >= q as u128 { (sum - q as u128) as u64 } else { sum as u64 }
                    })
                    .collect()
            })
            .collect();
        
        Self { limbs, n: self.n }
    }
    
    /// Subtract two RNS polynomials
    pub fn sub(&self, other: &Self, ctx: &RNSContext) -> Self {
        assert_eq!(self.n, other.n);
        
        let limbs = self.limbs.iter()
            .zip(other.limbs.iter())
            .zip(ctx.primes.iter())
            .map(|((a, b), &q)| {
                a.iter().zip(b.iter())
                    .map(|(&ai, &bi)| {
                        if ai >= bi { ai - bi } else { q - bi + ai }
                    })
                    .collect()
            })
            .collect();
        
        Self { limbs, n: self.n }
    }
    
    /// Negate RNS polynomial
    pub fn neg(&self, ctx: &RNSContext) -> Self {
        let limbs = self.limbs.iter()
            .zip(ctx.primes.iter())
            .map(|(a, &q)| {
                a.iter().map(|&ai| if ai == 0 { 0 } else { q - ai }).collect()
            })
            .collect();
        
        Self { limbs, n: self.n }
    }
    
    /// Multiply two RNS polynomials using parallel NTT
    pub fn mul(&self, other: &Self, ctx: &RNSContext) -> Self {
        assert_eq!(self.n, other.n);
        
        let limbs = self.limbs.iter()
            .zip(other.limbs.iter())
            .zip(ctx.ntt_engines.iter())
            .map(|((a, b), ntt)| {
                ntt.multiply(a, b)
            })
            .collect();
        
        Self { limbs, n: self.n }
    }
    
    /// Drop the last prime (for rescaling)
    pub fn drop_last_prime(&self, ctx: &RNSContext) -> Self {
        assert!(ctx.num_primes() > 1);
        
        let limbs = self.limbs[..self.limbs.len() - 1].to_vec();
        Self { limbs, n: self.n }
    }
}

/// Modular inverse using extended Euclidean algorithm
fn mod_inverse(a: u64, m: u64) -> u64 {
    let mut mn = (m as i128, a as i128);
    let mut xy = (0i128, 1i128);
    
    while mn.1 != 0 {
        let q = mn.0 / mn.1;
        mn = (mn.1, mn.0 - q * mn.1);
        xy = (xy.1, xy.0 - q * xy.1);
    }
    
    while xy.0 < 0 {
        xy.0 += m as i128;
    }
    (xy.0 % m as i128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // For basic RNS integer tests, we don't need NTT.
    // Create a simpler context that skips NTT for non-polynomial tests.
    
    #[test]
    fn test_rns_roundtrip() {
        // Use NTT-compatible primes
        let primes = vec![998244353, 985661441];
        let ctx = RNSContext::new(primes, 4);
        
        for x in [0u64, 1, 100, 1000, 7000] {
            let rns = ctx.from_int(x);
            let back = ctx.to_int(&rns);
            assert_eq!(back, x as u128, "Roundtrip failed for {}", x);
        }
    }
    
    #[test]
    fn test_rns_add() {
        let primes = vec![998244353, 985661441];
        let ctx = RNSContext::new(primes, 4);
        
        let a = ctx.from_int(100);
        let b = ctx.from_int(200);
        let sum = ctx.add(&a, &b);
        let result = ctx.to_int(&sum);
        
        assert_eq!(result, 300);
    }
    
    #[test]
    fn test_rns_mul() {
        let primes = vec![998244353, 985661441];
        let ctx = RNSContext::new(primes, 4);
        
        let a = ctx.from_int(12);
        let b = ctx.from_int(34);
        let prod = ctx.mul(&a, &b);
        let result = ctx.to_int(&prod);
        
        assert_eq!(result, 408);
    }
    
    #[test]
    fn test_rns_polynomial_add() {
        let primes = vec![998244353, 985661441];
        let ctx = RNSContext::new(primes, 4);
        
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        
        let rns_a = RNSPolynomial::from_poly(&a, &ctx);
        let rns_b = RNSPolynomial::from_poly(&b, &ctx);
        
        let rns_sum = rns_a.add(&rns_b, &ctx);
        
        // Check first limb
        assert_eq!(rns_sum.limbs[0], vec![6, 8, 10, 12]);
    }
    
    #[test]
    fn test_rns_polynomial_mul() {
        let primes = vec![998244353, 985661441];
        let ctx = RNSContext::new(primes, 8);
        
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let a = vec![1, 2, 0, 0, 0, 0, 0, 0];
        let b = vec![3, 4, 0, 0, 0, 0, 0, 0];
        
        let rns_a = RNSPolynomial::from_poly(&a, &ctx);
        let rns_b = RNSPolynomial::from_poly(&b, &ctx);
        
        let rns_prod = rns_a.mul(&rns_b, &ctx);
        
        // Check both limbs have correct result
        assert_eq!(rns_prod.limbs[0][0], 3);
        assert_eq!(rns_prod.limbs[0][1], 10);
        assert_eq!(rns_prod.limbs[0][2], 8);
        
        assert_eq!(rns_prod.limbs[1][0], 3);
        assert_eq!(rns_prod.limbs[1][1], 10);
        assert_eq!(rns_prod.limbs[1][2], 8);
    }
}
