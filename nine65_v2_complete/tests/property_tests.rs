//! Property-Based Tests for QMNF FHE
//!
//! Uses proptest to verify algebraic properties hold for all inputs.

use proptest::prelude::*;
use qmnf_fhe::prelude::*;
use qmnf_fhe::ops::encrypt::{BFVEncoder, BFVEncryptor, BFVDecryptor};
use qmnf_fhe::ops::homomorphic::BFVEvaluator;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Generate plaintext values in valid range
fn plaintext_strategy(t: u64) -> impl Strategy<Value = u64> {
    0..t
}

/// Generate small plaintext values for multiplication tests
fn small_plaintext_strategy(t: u64) -> impl Strategy<Value = u64> {
    // Keep products small to avoid wrapping issues
    0..(t.min(256))
}

// =============================================================================
// ENCRYPT/DECRYPT PROPERTIES
// =============================================================================

proptest! {
    /// Property: Encrypt then decrypt returns the original value
    #[test]
    fn prop_encrypt_decrypt_roundtrip(m in 0u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        
        let ct = encryptor.encrypt(m, &mut harvester);
        let result = decryptor.decrypt(&ct);
        
        prop_assert_eq!(result, m, "Decrypt(Encrypt(m)) should equal m");
    }
    
    /// Property: Different seeds produce different ciphertexts (with high probability)
    #[test]
    fn prop_encrypt_randomized(m in 0u64..2053, seed1 in 1u64..1000, seed2 in 1001u64..2000) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut h_keys = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut h_keys);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        
        let mut h1 = ShadowHarvester::with_seed(seed1);
        let mut h2 = ShadowHarvester::with_seed(seed2);
        
        let ct1 = encryptor.encrypt(m, &mut h1);
        let ct2 = encryptor.encrypt(m, &mut h2);
        
        // Ciphertexts should differ even for same plaintext
        prop_assert_ne!(ct1.c0.coeffs, ct2.c0.coeffs, 
                       "Different seeds should produce different ciphertexts");
    }
}

// =============================================================================
// HOMOMORPHIC ADDITION PROPERTIES
// =============================================================================

proptest! {
    /// Property: Homomorphic addition is correct
    #[test]
    fn prop_homo_add_correct(a in 0u64..1000, b in 0u64..1000) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_b = encryptor.encrypt(b, &mut harvester);
        let ct_sum = evaluator.add(&ct_a, &ct_b);
        
        let result = decryptor.decrypt(&ct_sum);
        let expected = (a + b) % config.t;
        
        prop_assert_eq!(result, expected, "Dec(Enc(a) + Enc(b)) should equal a + b mod t");
    }
    
    /// Property: Homomorphic addition is commutative
    #[test]
    fn prop_homo_add_commutative(a in 0u64..1000, b in 0u64..1000) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_b = encryptor.encrypt(b, &mut harvester);
        
        let ct_ab = evaluator.add(&ct_a, &ct_b);
        let ct_ba = evaluator.add(&ct_b, &ct_a);
        
        let result_ab = decryptor.decrypt(&ct_ab);
        let result_ba = decryptor.decrypt(&ct_ba);
        
        prop_assert_eq!(result_ab, result_ba, "a + b should equal b + a");
    }
    
    /// Property: Adding zero is identity
    #[test]
    fn prop_homo_add_zero_identity(a in 0u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_zero = encryptor.encrypt(0, &mut harvester);
        let ct_sum = evaluator.add(&ct_a, &ct_zero);
        
        let result = decryptor.decrypt(&ct_sum);
        
        prop_assert_eq!(result, a, "a + 0 should equal a");
    }
}

// =============================================================================
// PLAINTEXT OPERATION PROPERTIES
// =============================================================================

proptest! {
    /// Property: Adding plaintext is correct
    #[test]
    fn prop_add_plain_correct(a in 0u64..1000, b in 0u64..1000) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_sum = evaluator.add_plain(&ct_a, b);
        
        let result = decryptor.decrypt(&ct_sum);
        let expected = (a + b) % config.t;
        
        prop_assert_eq!(result, expected, "Dec(Enc(a) + b) should equal a + b mod t");
    }
    
    /// Property: Multiplying by plaintext is correct
    #[test]
    fn prop_mul_plain_correct(a in 0u64..100, b in 0u64..20) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_prod = evaluator.mul_plain(&ct_a, b);
        
        let result = decryptor.decrypt(&ct_prod);
        let expected = (a * b) % config.t;
        
        prop_assert_eq!(result, expected, "Dec(Enc(a) * b) should equal a * b mod t");
    }
    
    /// Property: Multiplying by 1 is identity
    #[test]
    fn prop_mul_plain_one_identity(a in 0u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_prod = evaluator.mul_plain(&ct_a, 1);
        
        let result = decryptor.decrypt(&ct_prod);
        
        prop_assert_eq!(result, a, "a * 1 should equal a");
    }
    
    /// Property: Multiplying by 0 gives 0
    #[test]
    fn prop_mul_plain_zero_absorbing(a in 0u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_prod = evaluator.mul_plain(&ct_a, 0);
        
        let result = decryptor.decrypt(&ct_prod);
        
        prop_assert_eq!(result, 0, "a * 0 should equal 0");
    }
}

// =============================================================================
// NEGATION AND SUBTRACTION PROPERTIES
// =============================================================================

proptest! {
    /// Property: Negation followed by addition gives 0
    #[test]
    fn prop_negate_add_zero(a in 1u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_neg_a = evaluator.negate(&ct_a);
        let ct_sum = evaluator.add(&ct_a, &ct_neg_a);
        
        let result = decryptor.decrypt(&ct_sum);
        
        prop_assert_eq!(result, 0, "a + (-a) should equal 0");
    }
    
    /// Property: Subtraction is correct
    #[test]
    fn prop_subtraction_correct(a in 500u64..2053, b in 0u64..500) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        let ct_a = encryptor.encrypt(a, &mut harvester);
        let ct_b = encryptor.encrypt(b, &mut harvester);
        let ct_diff = evaluator.sub(&ct_a, &ct_b);
        
        let result = decryptor.decrypt(&ct_diff);
        let expected = (a - b) % config.t;
        
        prop_assert_eq!(result, expected, "Dec(Enc(a) - Enc(b)) should equal a - b mod t");
    }
}

// =============================================================================
// SECURE KEY GENERATION PROPERTIES
// =============================================================================

proptest! {
    /// Property: Secure keygen is non-deterministic
    #[test]
    fn prop_secure_keygen_random(_seed in 0u64..1000) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        
        let keys1 = KeySet::generate_secure(&config, &ntt);
        let keys2 = KeySet::generate_secure(&config, &ntt);
        
        // Keys should be different (probability of collision is negligible)
        prop_assert_ne!(&keys1.secret_key.s.coeffs, &keys2.secret_key.s.coeffs,
                       "Secure keygen should produce different keys");
    }
    
    /// Property: Secure keys work for encryption
    #[test]
    fn prop_secure_keys_work(m in 0u64..2053) {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        
        let keys = KeySet::generate_secure(&config, &ntt);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        
        let mut harvester = ShadowHarvester::from_os_seed();
        let ct = encryptor.encrypt(m, &mut harvester);
        let result = decryptor.decrypt(&ct);
        
        prop_assert_eq!(result, m, "Secure keys should work for encrypt/decrypt");
    }
}

// =============================================================================
// NTT PROPERTIES
// =============================================================================

proptest! {
    /// Property: NTT is invertible
    #[test]
    fn prop_ntt_invertible(coeffs in prop::collection::vec(0u64..998244353, 1024)) {
        let q = 998244353u64;
        let n = 1024usize;
        let ntt = NTTEngine::new(q, n);
        
        let ntt_form = ntt.ntt(&coeffs);
        let recovered = ntt.intt(&ntt_form);
        
        prop_assert_eq!(&recovered, &coeffs, "iNTT(NTT(x)) should equal x");
    }
}
