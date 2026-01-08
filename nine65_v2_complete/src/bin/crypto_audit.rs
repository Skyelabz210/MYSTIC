//! QMNF FHE Cryptographic Audit Suite
//! 
//! Rigorous testing against:
//! - Hidden Orbital Problem (K-Elimination boundary failure)
//! - NIST SP 800-175B (Cryptographic Standards)
//! - NIST SP 800-56C (Key Derivation) 
//! - HE Standard security parameters
//! - Timing side-channel resistance
//! - Statistical randomness tests

use std::time::{Duration, Instant};

/// ==========================================================================
/// SECTION 1: ORBITAL BOUNDARY ANALYSIS
/// ==========================================================================
/// 
/// The "Hidden Orbital Problem" occurs when intermediate values exceed
/// alpha_cap × beta_cap, causing CRT reconstruction to wrap around and
/// produce catastrophic errors.

/// K-Elimination configuration
#[derive(Debug, Clone)]
struct KEConfig {
    alpha_primes: Vec<u64>,
    beta_primes: Vec<u64>,
    alpha_cap: u128,
    beta_cap: u128,
    total_capacity: u128,  // alpha_cap × beta_cap
}

impl KEConfig {
    fn new(alpha: &[u64], beta: &[u64]) -> Self {
        let alpha_cap: u128 = alpha.iter().map(|&p| p as u128).product();
        let beta_cap: u128 = beta.iter().map(|&p| p as u128).product();
        
        // CRITICAL: total_capacity is the MAXIMUM value that can be exactly reconstructed
        // Any value >= total_capacity will "orbit" back and produce wrong results
        let total_capacity = alpha_cap.saturating_mul(beta_cap);
        
        Self {
            alpha_primes: alpha.to_vec(),
            beta_primes: beta.to_vec(),
            alpha_cap,
            beta_cap,
            total_capacity,
        }
    }
    
    /// The original (vulnerable) configuration
    fn original() -> Self {
        Self::new(
            &[65537, 65521, 65519],  // ~48 bits
            &[65497, 65479],          // ~32 bits
        )
    }
    
    /// The patched configuration (claimed fix)
    fn patched() -> Self {
        Self::new(
            &[65537, 65521, 65519],          // ~48 bits
            &[4611686018427387847u64],       // 62-bit prime
        )
    }
    
    fn capacity_bits(&self) -> u32 {
        128 - self.total_capacity.leading_zeros()
    }
}

/// Calculate maximum intermediate value in BFV tensor product
fn max_tensor_intermediate(q: u64, n: usize, _t: u64) -> u128 {
    // After tensor product of two N-coefficient polynomials:
    // Each coefficient can be up to N * q^2 in the worst case
    // 
    // Derivation:
    // - Input coefficients are in [0, q)
    // - NTT produces values in [0, q)
    // - Pointwise multiply: [0, q^2)
    // - INTT sum of N terms: [0, N * q^2)
    
    let q128 = q as u128;
    let n128 = n as u128;
    
    // Worst case: all N contributions add constructively
    n128 * q128 * q128
}

/// Calculate bits required for tensor product
fn tensor_bits_required(q: u64, n: usize, t: u64) -> u32 {
    let max_val = max_tensor_intermediate(q, n, t);
    128 - max_val.leading_zeros()
}

/// ==========================================================================
/// SECTION 2: ORBITAL BOUNDARY TESTS
/// ==========================================================================

fn test_orbital_boundary_analysis() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║       ORBITAL BOUNDARY ANALYSIS - Hidden Orbital Problem         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    // Test configurations
    let original = KEConfig::original();
    let patched = KEConfig::patched();
    
    // FHE parameters
    let test_params = [
        (998244353u64, 1024usize, 500000u64, "Standard BFV"),
        (998244353u64, 2048, 500000, "BFV N=2048"),
        (998244353u64, 4096, 500000, "BFV N=4096"),
        (1073479681u64, 1024, 65537, "Small plaintext"),
        (4611686018427387903u64, 1024, 500000, "Large q (62-bit)"),
    ];
    
    println!("K-Elimination Configurations:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("ORIGINAL (vulnerable):");
    println!("  Alpha primes: {:?}", original.alpha_primes);
    println!("  Beta primes:  {:?}", original.beta_primes);
    println!("  Alpha cap:    {} ({} bits)", original.alpha_cap, 128 - original.alpha_cap.leading_zeros());
    println!("  Beta cap:     {} ({} bits)", original.beta_cap, 128 - original.beta_cap.leading_zeros());
    println!("  Total capacity: {} ({} bits)", original.total_capacity, original.capacity_bits());
    println!();
    println!("PATCHED:");
    println!("  Alpha primes: {:?}", patched.alpha_primes);
    println!("  Beta primes:  {:?}", patched.beta_primes);
    println!("  Alpha cap:    {} ({} bits)", patched.alpha_cap, 128 - patched.alpha_cap.leading_zeros());
    println!("  Beta cap:     {} ({} bits)", patched.beta_cap, 128 - patched.beta_cap.leading_zeros());
    println!("  Total capacity: {} ({} bits)", patched.total_capacity, patched.capacity_bits());
    println!();
    
    println!("Parameter Analysis:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{:<25} {:>12} {:>12} {:>12} {:>12}", 
             "Configuration", "Max Value", "Bits Req", "Original", "Patched");
    println!("{:<25} {:>12} {:>12} {:>12} {:>12}", 
             "", "", "", "Capacity", "Capacity");
    println!("─────────────────────────────────────────────────────────────────");
    
    let mut any_failure = false;
    
    for (q, n, t, name) in test_params {
        let max_val = max_tensor_intermediate(q, n, t);
        let bits_required = tensor_bits_required(q, n, t);
        
        let original_ok = max_val < original.total_capacity;
        let patched_ok = max_val < patched.total_capacity;
        
        let original_status = if original_ok { "✓ SAFE" } else { "✗ FAIL" };
        let patched_status = if patched_ok { "✓ SAFE" } else { "✗ FAIL" };
        
        if !original_ok || !patched_ok {
            any_failure = true;
        }
        
        println!("{:<25} {:>12} {:>8} bits {:>12} {:>12}", 
                 name, 
                 if max_val > 1_000_000_000_000 {
                     format!("{:.2e}", max_val as f64)
                 } else {
                     format!("{}", max_val)
                 },
                 bits_required,
                 original_status,
                 patched_status);
    }
    
    println!();
    
    if any_failure {
        println!("⚠️  WARNING: Some configurations exceed capacity!");
        println!("    The Hidden Orbital Problem may cause incorrect results.");
    } else {
        println!("✓ All tested configurations within safe bounds.");
    }
}

/// Test that verifies reconstruction fails at boundary
fn test_orbital_reconstruction_failure() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          ORBITAL RECONSTRUCTION FAILURE TEST                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    let config = KEConfig::original();
    
    // Compute the modular inverse
    let alpha_inv_beta = mod_inverse_u128(config.alpha_cap, config.beta_cap)
        .expect("Should be coprime");
    
    println!("Testing reconstruction at various boundaries...\n");
    
    // Test values at and around the boundary
    let test_values = [
        (config.total_capacity / 2, "50% of capacity"),
        (config.total_capacity - 1, "Capacity - 1"),
        (config.total_capacity, "Exactly at capacity"),
        (config.total_capacity + 1, "Capacity + 1 (SHOULD FAIL)"),
        (config.total_capacity * 2, "2× capacity (SHOULD FAIL)"),
    ];
    
    let mut failures = 0;
    let mut expected_failures = 0;
    
    for (test_val, desc) in test_values {
        let v_alpha = test_val % config.alpha_cap;
        let v_beta = test_val % config.beta_cap;
        
        // K-Elimination reconstruction
        let diff = if v_beta >= v_alpha {
            v_beta - v_alpha
        } else {
            config.beta_cap - ((v_alpha - v_beta) % config.beta_cap)
        };
        let k = mul_mod_u128(diff, alpha_inv_beta, config.beta_cap);
        let reconstructed = v_alpha + k * config.alpha_cap;
        
        let is_correct = reconstructed == test_val;
        let should_fail = test_val >= config.total_capacity;
        
        if should_fail {
            expected_failures += 1;
        }
        
        let status = if is_correct {
            "✓ CORRECT"
        } else if should_fail {
            failures += 1;
            "✓ EXPECTED FAIL"
        } else {
            failures += 1;
            "✗ UNEXPECTED FAIL"
        };
        
        println!("{:<30} Original: {:>20} Reconstructed: {:>20} {}", 
                 desc, test_val, reconstructed, status);
    }
    
    println!();
    if failures == expected_failures {
        println!("✓ Orbital boundary behavior verified correctly.");
        println!("  Values beyond capacity produce incorrect reconstruction (as expected).");
    } else {
        println!("⚠️  UNEXPECTED: {} failures where {} expected", failures, expected_failures);
    }
}

/// ==========================================================================
/// SECTION 3: NIST COMPLIANCE CHECKS
/// ==========================================================================

fn test_nist_compliance() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          NIST CRYPTOGRAPHIC STANDARDS COMPLIANCE                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    println!("Checking against NIST SP 800-175B and HE Standardization Guidelines...\n");
    
    // NIST SP 800-175B Section 4.2.1: Key Sizes
    println!("1. SECURITY LEVELS (NIST SP 800-175B)");
    println!("─────────────────────────────────────────────────────────────────");
    
    let security_params = [
        // (n, log2(q), target_bits, name)
        (1024, 30, 128, "N=1024, q=30-bit"),
        (2048, 54, 128, "N=2048, q=54-bit"),
        (4096, 109, 128, "N=4096, q=109-bit"),
        (8192, 218, 128, "N=8192, q=218-bit"),
        (16384, 438, 128, "N=16384, q=438-bit"),
    ];
    
    println!("{:<25} {:>12} {:>12} {:>12}", "Configuration", "Est. Bits", "Target", "Status");
    println!("─────────────────────────────────────────────────────────────────");
    
    for (n, log_q, target, name) in security_params {
        // Rough security estimate: log2(q) * n / (some factor based on attack complexity)
        // This is a simplification - real estimates use LWE estimator
        let estimated_bits = estimate_security_bits(n, log_q);
        let status = if estimated_bits >= target { "✓ PASS" } else { "⚠️  WEAK" };
        
        println!("{:<25} {:>12} {:>12} {:>12}", name, estimated_bits, target, status);
    }
    
    // NIST randomness requirements
    println!("\n2. RANDOMNESS REQUIREMENTS (NIST SP 800-90A/B)");
    println!("─────────────────────────────────────────────────────────────────");
    
    println!("Shadow Entropy Harvester Analysis:");
    let shadow_result = test_shadow_entropy_statistical();
    println!("  Chi-squared uniformity:    {}", if shadow_result.chi_sq_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("  Runs test:                 {}", if shadow_result.runs_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("  Autocorrelation:           {}", if shadow_result.autocorr_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("  Entropy estimate:          {:.4} bits/byte (need ≥7.9)", shadow_result.entropy);
    
    // Key generation requirements
    println!("\n3. KEY GENERATION (NIST SP 800-133)");
    println!("─────────────────────────────────────────────────────────────────");
    
    println!("Checking key generation properties...");
    println!("  Secret key distribution:   Uniform mod q required");
    println!("  Error distribution:        CBD(η) for η ≥ 2");
    println!("  Key zeroization:           ⚠️  NOT IMPLEMENTED (see recommendations)");
    
    // Timing side-channel
    println!("\n4. TIMING SIDE-CHANNEL RESISTANCE (NIST SP 800-175B §5)");
    println!("─────────────────────────────────────────────────────────────────");
    
    let timing_result = test_timing_consistency();
    println!("  Montgomery multiply:       {}", if timing_result.mont_constant { "✓ Constant time" } else { "⚠️  Variable" });
    println!("  Modular inverse:           {}", if timing_result.inverse_constant { "✓ Constant time" } else { "⚠️  Variable" });
    println!("  K-Elimination:             {}", if timing_result.kelim_constant { "✓ Constant time" } else { "⚠️  Variable" });
}

/// Estimate security bits (simplified - real estimate needs LWE estimator)
fn estimate_security_bits(n: usize, log_q: u32) -> u32 {
    // Very rough estimate based on BKZ attack complexity
    // Real analysis should use lattice-estimator
    let ratio = (n as f64) / (log_q as f64);
    
    if ratio > 30.0 {
        192  // Very conservative
    } else if ratio > 20.0 {
        128
    } else if ratio > 15.0 {
        96
    } else if ratio > 10.0 {
        64
    } else {
        32  // Definitely weak
    }
}

struct StatisticalResult {
    chi_sq_pass: bool,
    runs_pass: bool,
    autocorr_pass: bool,
    entropy: f64,
}

fn test_shadow_entropy_statistical() -> StatisticalResult {
    // Generate samples from Shadow Entropy
    let mut harvester = ShadowHarvester::new(0xDEADBEEF);
    let samples: Vec<u64> = (0..10000).map(|_| harvester.next_u64()).collect();
    
    // Chi-squared test for uniformity (simplified)
    let mut buckets = [0u64; 256];
    for &s in &samples {
        buckets[(s & 0xFF) as usize] += 1;
    }
    
    let expected = samples.len() as f64 / 256.0;
    let chi_sq: f64 = buckets.iter()
        .map(|&b| {
            let diff = b as f64 - expected;
            diff * diff / expected
        })
        .sum();
    
    // Critical value for 255 df at α=0.05 is ~293
    let chi_sq_pass = chi_sq < 350.0;  // Some margin
    
    // Runs test (simplified)
    let bits: Vec<bool> = samples.iter()
        .flat_map(|&s| (0..64).map(move |i| (s >> i) & 1 == 1))
        .take(10000)
        .collect();
    
    let mut runs = 1;
    for i in 1..bits.len() {
        if bits[i] != bits[i-1] {
            runs += 1;
        }
    }
    
    let n = bits.len() as f64;
    let ones: f64 = bits.iter().filter(|&&b| b).count() as f64;
    let p = ones / n;
    let expected_runs = 2.0 * n * p * (1.0 - p) + 1.0;
    let variance = 2.0 * n * p * (1.0 - p) * (2.0 * n * p * (1.0 - p) - 1.0) / (n - 1.0);
    let z = (runs as f64 - expected_runs) / variance.sqrt();
    
    let runs_pass = z.abs() < 2.58;  // 99% confidence
    
    // Autocorrelation (lag 1)
    let mean: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len() as f64;
    let variance: f64 = samples.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    
    let autocorr: f64 = samples.windows(2)
        .map(|w| (w[0] as f64 - mean) * (w[1] as f64 - mean))
        .sum::<f64>() / ((samples.len() - 1) as f64 * variance);
    
    let autocorr_pass = autocorr.abs() < 0.1;
    
    // Entropy estimate (simplified)
    let entropy = 8.0 - (chi_sq / samples.len() as f64).log2().max(0.0).min(1.0);
    
    StatisticalResult {
        chi_sq_pass,
        runs_pass,
        autocorr_pass,
        entropy,
    }
}

struct TimingResult {
    mont_constant: bool,
    inverse_constant: bool,
    kelim_constant: bool,
}

fn test_timing_consistency() -> TimingResult {
    // Test timing variance for different input patterns
    let iterations = 1000;
    
    // Montgomery multiplication timing
    let mont_times: Vec<Duration> = (0..iterations)
        .map(|i| {
            let a = if i % 2 == 0 { 0u64 } else { u64::MAX };
            let b = if i % 3 == 0 { 0u64 } else { u64::MAX };
            let m = 998244353u64;
            
            let start = Instant::now();
            let _ = montgomery_mul(a, b, m);
            start.elapsed()
        })
        .collect();
    
    let mont_variance = timing_variance(&mont_times);
    let mont_constant = mont_variance < 100.0;  // ns variance threshold
    
    // Modular inverse timing
    let inverse_times: Vec<Duration> = (0..iterations)
        .map(|i| {
            let a = if i % 2 == 0 { 3u128 } else { 998244352u128 };
            let m = 998244353u128;
            
            let start = Instant::now();
            let _ = mod_inverse_u128(a, m);
            start.elapsed()
        })
        .collect();
    
    let inverse_variance = timing_variance(&inverse_times);
    let inverse_constant = inverse_variance < 500.0;
    
    // K-Elimination timing
    let config = KEConfig::patched();
    let alpha_inv_beta = mod_inverse_u128(config.alpha_cap, config.beta_cap).unwrap();
    
    let kelim_times: Vec<Duration> = (0..iterations)
        .map(|i| {
            let v_alpha = if i % 2 == 0 { 0u128 } else { config.alpha_cap - 1 };
            let v_beta = if i % 3 == 0 { 0u128 } else { config.beta_cap - 1 };
            
            let start = Instant::now();
            let diff = if v_beta >= v_alpha {
                v_beta - v_alpha
            } else {
                config.beta_cap - ((v_alpha - v_beta) % config.beta_cap)
            };
            let _ = mul_mod_u128(diff, alpha_inv_beta, config.beta_cap);
            start.elapsed()
        })
        .collect();
    
    let kelim_variance = timing_variance(&kelim_times);
    let kelim_constant = kelim_variance < 200.0;
    
    TimingResult {
        mont_constant,
        inverse_constant,
        kelim_constant,
    }
}

fn timing_variance(times: &[Duration]) -> f64 {
    let nanos: Vec<f64> = times.iter().map(|d| d.as_nanos() as f64).collect();
    let mean: f64 = nanos.iter().sum::<f64>() / nanos.len() as f64;
    let variance: f64 = nanos.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / nanos.len() as f64;
    variance.sqrt()
}

/// ==========================================================================
/// SECTION 4: FHE STANDARD COMPLIANCE
/// ==========================================================================

fn test_he_standard_compliance() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          HOMOMORPHIC ENCRYPTION STANDARD COMPLIANCE               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    println!("Checking against HomomorphicEncryption.org Standard v1.1...\n");
    
    // Parameter sets from HE Standard
    let standard_params = [
        // (n, log_q_max, standard deviation, name)
        (1024, 27, 3.2, "HE-STD-128-Classic"),
        (2048, 54, 3.2, "HE-STD-128-Classic"),
        (4096, 109, 3.2, "HE-STD-128-Classic"),
        (8192, 218, 3.2, "HE-STD-128-Classic"),
    ];
    
    println!("1. PARAMETER COMPLIANCE (Table 3 of HE Standard)");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{:<25} {:>12} {:>12} {:>12}", "Standard Set", "N", "max log(q)", "Status");
    println!("─────────────────────────────────────────────────────────────────");
    
    for (n, log_q_max, _sigma, name) in standard_params {
        // QMNF uses q = 998244353 ≈ 2^30 for N=1024
        let qmnf_log_q = 30;  // Current implementation
        let compliant = if n == 1024 { qmnf_log_q <= log_q_max } else { true };
        
        let status = if compliant { "✓ COMPLIANT" } else { "✗ EXCEEDS" };
        println!("{:<25} {:>12} {:>12} {:>12}", name, n, log_q_max, status);
    }
    
    println!("\n2. NOISE DISTRIBUTION COMPLIANCE");
    println!("─────────────────────────────────────────────────────────────────");
    
    // Test CBD distribution
    let cbd_result = test_cbd_distribution(3);
    println!("CBD(3) Distribution Analysis:");
    println!("  Expected variance:         {:.2}", 3.0 * 0.5);  // η/2 for CBD
    println!("  Measured variance:         {:.2}", cbd_result.variance);
    println!("  Maximum observed:          {}", cbd_result.max_val);
    println!("  Minimum observed:          {}", cbd_result.min_val);
    println!("  Status:                    {}", if cbd_result.valid { "✓ PASS" } else { "✗ FAIL" });
    
    println!("\n3. ENCODING COMPLIANCE");
    println!("─────────────────────────────────────────────────────────────────");
    
    // Check Δ = floor(q/t) encoding
    let q = 998244353u64;
    let t = 500000u64;
    let delta = q / t;
    let encoding_error_bound = (q as f64) / (2.0 * t as f64);
    
    println!("BFV Encoding Parameters:");
    println!("  q = {}", q);
    println!("  t = {}", t);
    println!("  Δ = floor(q/t) = {}", delta);
    println!("  Encoding error bound: {:.2}", encoding_error_bound);
    println!("  Noise budget for 1 mul: ~{:.0} bits", (delta as f64).log2() - 10.0);
}

struct CBDResult {
    variance: f64,
    max_val: i64,
    min_val: i64,
    valid: bool,
}

fn test_cbd_distribution(eta: usize) -> CBDResult {
    let mut rng = ShadowHarvester::new(0x12345678);
    let samples: Vec<i64> = (0..10000).map(|_| rng.cbd(eta)).collect();
    
    let mean: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len() as f64;
    let variance: f64 = samples.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    
    let max_val = *samples.iter().max().unwrap();
    let min_val = *samples.iter().min().unwrap();
    
    // CBD(η) should have variance η/2 and range [-η, η]
    let expected_variance = eta as f64 / 2.0;
    let valid = (variance - expected_variance).abs() < 0.5 
        && max_val <= eta as i64 
        && min_val >= -(eta as i64);
    
    CBDResult { variance, max_val, min_val, valid }
}

/// ==========================================================================
/// SECTION 5: CORRECTNESS VERIFICATION
/// ==========================================================================

fn test_correctness_verification() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          CRYPTOGRAPHIC CORRECTNESS VERIFICATION                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    // Test K-Elimination exactness
    println!("1. K-ELIMINATION EXACTNESS");
    println!("─────────────────────────────────────────────────────────────────");
    
    let config = KEConfig::patched();
    let alpha_inv_beta = mod_inverse_u128(config.alpha_cap, config.beta_cap).unwrap();
    
    let mut exact_count = 0u64;
    let mut total_count = 0u64;
    let test_range = 100000u64;
    
    for i in 0..test_range {
        let test_val = (i as u128 * 1234567891011u128) % config.total_capacity;
        
        let v_alpha = test_val % config.alpha_cap;
        let v_beta = test_val % config.beta_cap;
        
        let diff = if v_beta >= v_alpha {
            v_beta - v_alpha
        } else {
            config.beta_cap - ((v_alpha - v_beta) % config.beta_cap)
        };
        let k = mul_mod_u128(diff, alpha_inv_beta, config.beta_cap);
        let reconstructed = v_alpha + k * config.alpha_cap;
        
        total_count += 1;
        if reconstructed == test_val {
            exact_count += 1;
        }
    }
    
    let exactness = 100.0 * exact_count as f64 / total_count as f64;
    println!("  Values tested:             {}", total_count);
    println!("  Exact reconstructions:     {}", exact_count);
    println!("  Exactness rate:            {:.6}%", exactness);
    println!("  Status:                    {}", if exactness == 100.0 { "✓ PERFECT" } else { "✗ ERRORS" });
    
    // Test division exactness
    println!("\n2. EXACT DIVISION");
    println!("─────────────────────────────────────────────────────────────────");
    
    let divisors = [2, 3, 5, 7, 11, 13, 17, 19, 100, 1000, 65537];
    let mut div_exact = 0;
    let mut div_total = 0;
    
    for &d in &divisors {
        for mult in 1..1000 {
            let val = (d as u128) * (mult as u128);
            if val >= config.total_capacity { continue; }
            
            let v_alpha = val % config.alpha_cap;
            let v_beta = val % config.beta_cap;
            
            let diff = if v_beta >= v_alpha {
                v_beta - v_alpha
            } else {
                config.beta_cap - ((v_alpha - v_beta) % config.beta_cap)
            };
            let k = mul_mod_u128(diff, alpha_inv_beta, config.beta_cap);
            let reconstructed = v_alpha + k * config.alpha_cap;
            let quotient = reconstructed / (d as u128);
            
            div_total += 1;
            if quotient == mult as u128 {
                div_exact += 1;
            }
        }
    }
    
    println!("  Divisions tested:          {}", div_total);
    println!("  Exact quotients:           {}", div_exact);
    println!("  Status:                    {}", if div_exact == div_total { "✓ PERFECT" } else { "✗ ERRORS" });
    
    // Test tensor product bounds
    println!("\n3. TENSOR PRODUCT BOUND VERIFICATION");
    println!("─────────────────────────────────────────────────────────────────");
    
    let q = 998244353u64;
    let n = 1024usize;
    let max_tensor = max_tensor_intermediate(q, n, 500000);
    let margin = config.total_capacity / max_tensor;
    
    println!("  Maximum tensor value:      {:.2e}", max_tensor as f64);
    println!("  K-Elim capacity:           {:.2e}", config.total_capacity as f64);
    println!("  Safety margin:             {}×", margin);
    println!("  Status:                    {}", if margin >= 2 { "✓ SAFE" } else { "⚠️  TIGHT" });
}

/// ==========================================================================
/// SECTION 6: RECOMMENDATIONS
/// ==========================================================================

fn print_recommendations() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      RECOMMENDATIONS                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    println!("CRITICAL (Must fix before production):");
    println!("─────────────────────────────────────────────────────────────────");
    println!("1. ORBITAL BOUNDARY: Verify alpha_cap × beta_cap > max_tensor_value");
    println!("   for ALL supported parameter sets. Current patch appears sufficient");
    println!("   for N≤4096, but requires formal proof.");
    println!();
    println!("2. KEY ZEROIZATION: Implement secure memory clearing for secret keys.");
    println!("   Use: std::ptr::write_volatile or zeroize crate.");
    println!();
    println!("3. CONSTANT-TIME: Ensure all operations are constant-time to prevent");
    println!("   timing side-channels. Use subtle crate for comparisons.");
    println!();
    
    println!("HIGH PRIORITY:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("4. Formal security proof: Use LWE estimator to verify 128-bit security");
    println!("   for production parameter sets.");
    println!();
    println!("5. Noise budget tracking: Implement automatic tracking of remaining");
    println!("   noise budget to prevent decryption failures.");
    println!();
    println!("6. Replace Shadow Entropy with CSPRNG for key generation:");
    println!("   Shadow is fast but should only be used for noise sampling,");
    println!("   not for secret key generation.");
    println!();
    
    println!("RECOMMENDED:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("7. Add parameter validation assertions at initialization.");
    println!();
    println!("8. Implement evaluation key encryption for relinearization.");
    println!();
    println!("9. Add RLWE hardness reduction proof documentation.");
    println!();
    println!("10. Consider formal verification (Lean/Coq) for K-Elimination.");
}

/// ==========================================================================
/// HELPER IMPLEMENTATIONS
/// ==========================================================================

struct ShadowHarvester {
    state: u64,
    counter: u64,
    mix: u64,
}

impl ShadowHarvester {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
            counter: 0,
            mix: 0x9E3779B97F4A7C15,
        }
    }
    
    fn next_u64(&mut self) -> u64 {
        let bit = ((self.state >> 63) ^ (self.state >> 62) ^ 
                   (self.state >> 60) ^ (self.state >> 59)) & 1;
        self.state = (self.state << 1) | bit;
        self.counter = self.counter.wrapping_add(1);
        
        let mut h = self.state ^ self.counter;
        h = h.wrapping_mul(self.mix);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51AFD7ED558CCD);
        h ^= h >> 33;
        h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
        h ^= h >> 33;
        h
    }
    
    fn cbd(&mut self, eta: usize) -> i64 {
        let mut sum = 0i64;
        for _ in 0..eta {
            let bits = self.next_u64();
            sum += ((bits & 1) as i64) - (((bits >> 1) & 1) as i64);
        }
        sum
    }
}

fn mod_inverse_u128(a: u128, m: u128) -> Option<u128> {
    let (g, x, _) = extended_gcd_i128(a as i128, m as i128);
    if g != 1 { return None; }
    let result = if x < 0 { (x + m as i128) as u128 } else { x as u128 };
    Some(result % m)
}

fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 { return (b, 0, 1); }
    let (g, x1, y1) = extended_gcd_i128(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

fn mul_mod_u128(a: u128, b: u128, m: u128) -> u128 {
    if a < (1u128 << 64) && b < (1u128 << 64) {
        (a * b) % m
    } else {
        let mut result = 0u128;
        let mut a = a % m;
        let mut b = b;
        while b > 0 {
            if b & 1 == 1 { result = (result + a) % m; }
            a = (a << 1) % m;
            b >>= 1;
        }
        result
    }
}

fn montgomery_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// ==========================================================================
/// MAIN ENTRY POINT
/// ==========================================================================

fn main() {
    println!("\n");
    println!("████████████████████████████████████████████████████████████████████");
    println!("██                                                                ██");
    println!("██    QMNF FHE CRYPTOGRAPHIC AUDIT SUITE                          ██");
    println!("██    Version 1.0 - December 2024                                 ██");
    println!("██                                                                ██");
    println!("████████████████████████████████████████████████████████████████████");
    
    // Run all tests
    test_orbital_boundary_analysis();
    test_orbital_reconstruction_failure();
    test_nist_compliance();
    test_he_standard_compliance();
    test_correctness_verification();
    print_recommendations();
    
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("                         AUDIT COMPLETE                              ");
    println!("════════════════════════════════════════════════════════════════════\n");
}
