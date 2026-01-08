//! Property-Based Tests for QMNF FHE
//!
//! Uses proptest to verify FHE properties hold for randomly generated inputs.

use proptest::prelude::*;
use qmnf_fhe::prelude::*;
use qmnf_fhe::params::FHEConfig;

/// Generate valid plaintext values for light config (t = 2053)
fn plaintext_light() -> impl Strategy<Value = u64> {
    0u64..2053u64
}

/// Generate small multipliers for mul_plain tests
fn small_multiplier() -> impl Strategy<Value = u64> {
    1u64..100u64
}

/// Generate random seeds for deterministic testing
fn random_seed() -> impl Strategy<Value = u64> {
    any::<u64>()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]
    
    /// Property: Encrypt then decrypt recovers original plaintext
    #[test]
    fn prop_encrypt_decrypt_roundtrip(
        m in plaintext_light(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        
        let ct = encryptor.encrypt(m, &mut harvester);
        let result = decryptor.decrypt(&ct);
        
        prop_assert_eq!(result, m, "Encrypt/decrypt roundtrip failed");
    }
    
    /// Property: Homomorphic addition produces correct result
    #[test]
    fn prop_homomorphic_add(
        m1 in plaintext_light(),
        m2 in plaintext_light(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct1 = encryptor.encrypt(m1, &mut harvester);
        let ct2 = encryptor.encrypt(m2, &mut harvester);
        let ct_sum = evaluator.add(&ct1, &ct2);
        let result = decryptor.decrypt(&ct_sum);
        
        let expected = (m1 + m2) % config.t;
        prop_assert_eq!(result, expected, 
                       "Homomorphic add failed: {} + {} = {} (expected {})",
                       m1, m2, result, expected);
    }
    
    /// Property: Add is commutative (ct1 + ct2 == ct2 + ct1)
    #[test]
    fn prop_add_commutative(
        m1 in plaintext_light(),
        m2 in plaintext_light(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct1 = encryptor.encrypt(m1, &mut harvester);
        let ct2 = encryptor.encrypt(m2, &mut harvester);
        
        let sum_12 = evaluator.add(&ct1, &ct2);
        let sum_21 = evaluator.add(&ct2, &ct1);
        
        let result_12 = decryptor.decrypt(&sum_12);
        let result_21 = decryptor.decrypt(&sum_21);
        
        prop_assert_eq!(result_12, result_21, "Addition should be commutative");
    }
    
    /// Property: Multiply by plaintext produces correct result
    #[test]
    fn prop_mul_plain(
        m in 0u64..100u64,
        k in small_multiplier(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct = encryptor.encrypt(m, &mut harvester);
        let ct_prod = evaluator.mul_plain(&ct, k);
        let result = decryptor.decrypt(&ct_prod);
        
        let expected = (m * k) % config.t;
        prop_assert_eq!(result, expected,
                       "Mul plain failed: {} * {} = {} (expected {})",
                       m, k, result, expected);
    }
    
    /// Property: Add plaintext produces correct result
    #[test]
    fn prop_add_plain(
        m in plaintext_light(),
        k in 0u64..1000u64,
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct = encryptor.encrypt(m, &mut harvester);
        let ct_sum = evaluator.add_plain(&ct, k);
        let result = decryptor.decrypt(&ct_sum);
        
        let expected = (m + k) % config.t;
        prop_assert_eq!(result, expected,
                       "Add plain failed: {} + {} = {} (expected {})",
                       m, k, result, expected);
    }
    
    /// Property: Double negation returns original
    #[test]
    fn prop_double_negate(
        m in plaintext_light(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct = encryptor.encrypt(m, &mut harvester);
        let ct_neg = evaluator.negate(&ct);
        let ct_double_neg = evaluator.negate(&ct_neg);
        let result = decryptor.decrypt(&ct_double_neg);
        
        prop_assert_eq!(result, m, "Double negation should return original");
    }
    
    /// Property: Subtraction is correct
    #[test]
    fn prop_subtraction(
        m1 in plaintext_light(),
        m2 in plaintext_light(),
        seed in random_seed()
    ) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(seed);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct1 = encryptor.encrypt(m1, &mut harvester);
        let ct2 = encryptor.encrypt(m2, &mut harvester);
        let ct_diff = evaluator.sub(&ct1, &ct2);
        let result = decryptor.decrypt(&ct_diff);
        
        let expected = if m1 >= m2 {
            m1 - m2
        } else {
            config.t - (m2 - m1)
        };
        
        prop_assert_eq!(result, expected,
                       "Subtraction failed: {} - {} = {} (expected {})",
                       m1, m2, result, expected);
    }
}

// Security property tests
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    
    /// Property: Security estimate is consistent across calls
    #[test]
    fn prop_security_consistent(
        n in prop::sample::select(vec![1024usize, 2048, 4096]),
        log_q in 20u32..55u32
    ) {
        use qmnf_fhe::security::LWEParams;
        
        let params = LWEParams::new(n, log_q, 3.2);
        let est1 = params.he_standard_estimate();
        let est2 = params.he_standard_estimate();
        
        prop_assert_eq!(est1.classical_bits, est2.classical_bits,
                       "Security estimate should be deterministic");
    }
}
