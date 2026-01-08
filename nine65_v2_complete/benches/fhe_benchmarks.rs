//! QMNF FHE Criterion Benchmarks
//!
//! Comprehensive benchmarking suite for all FHE operations.
//!
//! Run with:
//!   cargo bench
//!   cargo bench -- --save-baseline production_v1
//!   cargo bench -- --baseline production_v1

use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput,
};

use qmnf_fhe::prelude::*;
use qmnf_fhe::params::FHEConfig;

/// Benchmark key generation
fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("keygen");
    
    // Light config (N=1024)
    group.bench_function("light_deterministic", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        
        b.iter(|| {
            let mut harvester = ShadowHarvester::with_seed(42);
            black_box(KeySet::generate(&config, &ntt, &mut harvester))
        });
    });
    
    group.bench_function("light_secure", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        
        b.iter(|| {
            black_box(KeySet::generate_secure(&config, &ntt))
        });
    });
    
    // HE Standard config (N=2048)
    group.bench_function("he_standard_128_secure", |b| {
        let config = FHEConfig::he_standard_128();
        let ntt = NTTEngine::new(config.q, config.n);
        
        b.iter(|| {
            black_box(KeySet::generate_secure(&config, &ntt))
        });
    });
    
    group.finish();
}

/// Benchmark encryption
fn bench_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("encrypt");
    
    // Setup
    let config = FHEConfig::light();
    let ntt = NTTEngine::new(config.q, config.n);
    let mut setup_harvester = ShadowHarvester::with_seed(1);
    let keys = KeySet::generate(&config, &ntt, &mut setup_harvester);
    let encoder = BFVEncoder::new(&config);
    let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
    
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("light", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| {
            black_box(encryptor.encrypt(42, &mut harvester))
        });
    });
    
    // Batch encryption throughput
    group.bench_function("light_batch_100", |b| {
        b.iter(|| {
            let mut harvester = ShadowHarvester::with_seed(42);
            for i in 0..100 {
                black_box(encryptor.encrypt(i, &mut harvester));
            }
        });
    });
    
    group.finish();
}

/// Benchmark decryption
fn bench_decrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("decrypt");
    
    // Setup
    let config = FHEConfig::light();
    let ntt = NTTEngine::new(config.q, config.n);
    let mut harvester = ShadowHarvester::with_seed(1);
    let keys = KeySet::generate(&config, &ntt, &mut harvester);
    let encoder = BFVEncoder::new(&config);
    let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
    let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
    
    let ct = encryptor.encrypt(42, &mut harvester);
    
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("light", |b| {
        b.iter(|| {
            black_box(decryptor.decrypt(&ct))
        });
    });
    
    group.finish();
}

/// Benchmark homomorphic operations
fn bench_homomorphic(c: &mut Criterion) {
    let mut group = c.benchmark_group("homomorphic");
    
    // Setup
    let config = FHEConfig::light();
    let ntt = NTTEngine::new(config.q, config.n);
    let mut harvester = ShadowHarvester::with_seed(1);
    let keys = KeySet::generate(&config, &ntt, &mut harvester);
    let encoder = BFVEncoder::new(&config);
    let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
    let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
    
    let ct1 = encryptor.encrypt(17, &mut harvester);
    let ct2 = encryptor.encrypt(25, &mut harvester);
    
    group.throughput(Throughput::Elements(1));
    
    // Ciphertext addition
    group.bench_function("add_ct_ct", |b| {
        b.iter(|| {
            black_box(evaluator.add(&ct1, &ct2))
        });
    });
    
    // Ciphertext subtraction
    group.bench_function("sub_ct_ct", |b| {
        b.iter(|| {
            black_box(evaluator.sub(&ct1, &ct2))
        });
    });
    
    // Plaintext addition
    group.bench_function("add_plain", |b| {
        b.iter(|| {
            black_box(evaluator.add_plain(&ct1, 10))
        });
    });
    
    // Plaintext multiplication
    group.bench_function("mul_plain", |b| {
        b.iter(|| {
            black_box(evaluator.mul_plain(&ct1, 3))
        });
    });
    
    // Negation
    group.bench_function("negate", |b| {
        b.iter(|| {
            black_box(evaluator.negate(&ct1))
        });
    });
    
    // Ciphertext multiplication (no relin)
    group.bench_function("mul_ct_ct_no_relin", |b| {
        b.iter(|| {
            black_box(evaluator.mul(&ct1, &ct2))
        });
    });
    
    group.finish();
}

/// Benchmark NTT operations
fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    
    let configs = [
        ("n_1024", 1024usize, 998244353u64),
        ("n_2048", 2048, 998244353),
        ("n_4096", 4096, 998244353),
    ];
    
    for (name, n, q) in configs {
        let ntt = NTTEngine::new(q, n);
        let data: Vec<u64> = (0..n as u64).collect();
        
        group.throughput(Throughput::Elements(n as u64));
        
        group.bench_with_input(BenchmarkId::new("forward", name), &data, |b, data| {
            b.iter(|| {
                black_box(ntt.forward(data))
            });
        });
        
        let ntt_data = ntt.forward(&data);
        group.bench_with_input(BenchmarkId::new("inverse", name), &ntt_data, |b, ntt_data| {
            b.iter(|| {
                black_box(ntt.inverse(ntt_data))
            });
        });
    }
    
    group.finish();
}

/// Benchmark Montgomery arithmetic
fn bench_montgomery(c: &mut Criterion) {
    let mut group = c.benchmark_group("montgomery");
    
    let q = 998244353u64;
    let mont = MontgomeryContext::new(q);
    let a = 123456789u64;
    let b = 987654321u64 % q;
    
    group.bench_function("mul", |b_iter| {
        b_iter.iter(|| {
            black_box(mont.mul(a, b))
        });
    });
    
    group.bench_function("reduce", |b_iter| {
        let value = (a as u128) * (b as u128);
        b_iter.iter(|| {
            black_box(mont.reduce(value))
        });
    });
    
    group.finish();
}

/// Benchmark entropy generation
fn bench_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy");
    
    // Shadow Entropy
    group.bench_function("shadow_u64", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| {
            black_box(harvester.next_u64())
        });
    });
    
    // Secure entropy (OS CSPRNG)
    group.bench_function("secure_u64", |b| {
        use qmnf_fhe::entropy::secure_u64;
        b.iter(|| {
            black_box(secure_u64())
        });
    });
    
    // Ternary generation
    group.bench_function("shadow_ternary_1024", |b| {
        b.iter(|| {
            let mut harvester = ShadowHarvester::with_seed(42);
            let mut result = vec![0i64; 1024];
            for r in &mut result {
                *r = harvester.ternary();
            }
            black_box(result)
        });
    });
    
    group.bench_function("secure_ternary_1024", |b| {
        use qmnf_fhe::entropy::secure_ternary_vector;
        b.iter(|| {
            black_box(secure_ternary_vector(1024))
        });
    });
    
    group.finish();
}

/// Benchmark security estimation
fn bench_security(c: &mut Criterion) {
    let mut group = c.benchmark_group("security");
    
    group.bench_function("lwe_estimate", |b| {
        use qmnf_fhe::security::LWEParams;
        let params = LWEParams::new(2048, 54, 3.2);
        b.iter(|| {
            black_box(params.he_standard_estimate())
        });
    });
    
    group.finish();
}

/// Benchmark end-to-end operations
fn bench_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e");
    group.sample_size(50);  // Reduce samples for slow operations
    
    // Full encrypt + decrypt cycle
    group.bench_function("encrypt_decrypt_light", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(1);
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        
        b.iter(|| {
            let mut h = ShadowHarvester::with_seed(42);
            let ct = encryptor.encrypt(42, &mut h);
            black_box(decryptor.decrypt(&ct))
        });
    });
    
    // Encrypt + add + decrypt
    group.bench_function("encrypt_add_decrypt", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(1);
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        let encryptor = BFVEncryptor::new(&keys.public_key, &encoder, &ntt, config.eta);
        let decryptor = BFVDecryptor::new(&keys.secret_key, &encoder, &ntt);
        let evaluator = BFVEvaluator::new(&ntt, &encoder, Some(&keys.eval_key));
        
        b.iter(|| {
            let mut h = ShadowHarvester::with_seed(42);
            let ct1 = encryptor.encrypt(17, &mut h);
            let ct2 = encryptor.encrypt(25, &mut h);
            let ct_sum = evaluator.add(&ct1, &ct2);
            black_box(decryptor.decrypt(&ct_sum))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_keygen,
    bench_encrypt,
    bench_decrypt,
    bench_homomorphic,
    bench_ntt,
    bench_montgomery,
    bench_entropy,
    bench_security,
    bench_e2e,
);

criterion_main!(benches);
