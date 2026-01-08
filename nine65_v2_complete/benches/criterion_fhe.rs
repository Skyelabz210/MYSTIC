//! QMNF FHE Benchmarks
//!
//! Run with: cargo bench --bench fhe_benchmarks
//!
//! Results saved to target/criterion/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qmnf_fhe::prelude::*;
use qmnf_fhe::ops::encrypt::{BFVEncoder, BFVEncryptor, BFVDecryptor};
use qmnf_fhe::ops::homomorphic::BFVEvaluator;

/// Setup struct for benchmarks
struct BenchSetup {
    config: FHEConfig,
    ntt: NTTEngine,
    keys: KeySet,
    encoder: BFVEncoder,
}

impl BenchSetup {
    fn light() -> Self {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(0xBEEF_CAFE);
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        Self { config, ntt, keys, encoder }
    }
    
    fn he_standard_128() -> Self {
        let config = FHEConfig::he_standard_128();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(0xBEEF_CAFE);
        let keys = KeySet::generate(&config, &ntt, &mut harvester);
        let encoder = BFVEncoder::new(&config);
        Self { config, ntt, keys, encoder }
    }
}

// =============================================================================
// KEY GENERATION BENCHMARKS
// =============================================================================

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("keygen");
    
    // Light config (N=1024)
    group.bench_function("light_N1024", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        b.iter(|| {
            KeySet::generate(black_box(&config), black_box(&ntt), black_box(&mut harvester))
        });
    });
    
    // HE Standard 128 (N=2048)
    group.bench_function("he_standard_N2048", |b| {
        let config = FHEConfig::he_standard_128();
        let ntt = NTTEngine::new(config.q, config.n);
        let mut harvester = ShadowHarvester::with_seed(42);
        
        b.iter(|| {
            KeySet::generate(black_box(&config), black_box(&ntt), black_box(&mut harvester))
        });
    });
    
    // Secure keygen (CSPRNG)
    group.bench_function("secure_light_N1024", |b| {
        let config = FHEConfig::light();
        let ntt = NTTEngine::new(config.q, config.n);
        
        b.iter(|| {
            KeySet::generate_secure(black_box(&config), black_box(&ntt))
        });
    });
    
    group.finish();
}

// =============================================================================
// ENCRYPTION BENCHMARKS
// =============================================================================

fn bench_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("encrypt");
    
    let setup = BenchSetup::light();
    let encryptor = BFVEncryptor::new(&setup.keys.public_key, &setup.encoder, &setup.ntt, setup.config.eta);
    
    group.bench_function("light_N1024", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| {
            encryptor.encrypt(black_box(42), black_box(&mut harvester))
        });
    });
    
    // HE Standard 128
    let setup_128 = BenchSetup::he_standard_128();
    let encryptor_128 = BFVEncryptor::new(&setup_128.keys.public_key, &setup_128.encoder, &setup_128.ntt, setup_128.config.eta);
    
    group.bench_function("he_standard_N2048", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| {
            encryptor_128.encrypt(black_box(42), black_box(&mut harvester))
        });
    });
    
    group.finish();
}

// =============================================================================
// DECRYPTION BENCHMARKS
// =============================================================================

fn bench_decrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("decrypt");
    
    let setup = BenchSetup::light();
    let encryptor = BFVEncryptor::new(&setup.keys.public_key, &setup.encoder, &setup.ntt, setup.config.eta);
    let decryptor = BFVDecryptor::new(&setup.keys.secret_key, &setup.encoder, &setup.ntt);
    
    let mut harvester = ShadowHarvester::with_seed(42);
    let ct = encryptor.encrypt(42, &mut harvester);
    
    group.bench_function("light_N1024", |b| {
        b.iter(|| {
            decryptor.decrypt(black_box(&ct))
        });
    });
    
    // HE Standard 128
    let setup_128 = BenchSetup::he_standard_128();
    let encryptor_128 = BFVEncryptor::new(&setup_128.keys.public_key, &setup_128.encoder, &setup_128.ntt, setup_128.config.eta);
    let decryptor_128 = BFVDecryptor::new(&setup_128.keys.secret_key, &setup_128.encoder, &setup_128.ntt);
    
    let ct_128 = encryptor_128.encrypt(42, &mut harvester);
    
    group.bench_function("he_standard_N2048", |b| {
        b.iter(|| {
            decryptor_128.decrypt(black_box(&ct_128))
        });
    });
    
    group.finish();
}

// =============================================================================
// HOMOMORPHIC OPERATION BENCHMARKS
// =============================================================================

fn bench_homo_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("homo_add");
    
    let setup = BenchSetup::light();
    let encryptor = BFVEncryptor::new(&setup.keys.public_key, &setup.encoder, &setup.ntt, setup.config.eta);
    let evaluator = BFVEvaluator::new(&setup.ntt, &setup.encoder, Some(&setup.keys.eval_key));
    
    let mut harvester = ShadowHarvester::with_seed(42);
    let ct_a = encryptor.encrypt(17, &mut harvester);
    let ct_b = encryptor.encrypt(25, &mut harvester);
    
    group.bench_function("light_N1024", |b| {
        b.iter(|| {
            evaluator.add(black_box(&ct_a), black_box(&ct_b))
        });
    });
    
    group.finish();
}

fn bench_homo_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("homo_mul");
    
    let setup = BenchSetup::light();
    let encryptor = BFVEncryptor::new(&setup.keys.public_key, &setup.encoder, &setup.ntt, setup.config.eta);
    let evaluator = BFVEvaluator::new(&setup.ntt, &setup.encoder, Some(&setup.keys.eval_key));
    
    let mut harvester = ShadowHarvester::with_seed(42);
    let ct_a = encryptor.encrypt(17, &mut harvester);
    let ct_b = encryptor.encrypt(25, &mut harvester);
    
    group.bench_function("light_N1024", |b| {
        b.iter(|| {
            evaluator.mul(black_box(&ct_a), black_box(&ct_b))
        });
    });
    
    group.finish();
}

fn bench_mul_plain(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_plain");
    
    let setup = BenchSetup::light();
    let encryptor = BFVEncryptor::new(&setup.keys.public_key, &setup.encoder, &setup.ntt, setup.config.eta);
    let evaluator = BFVEvaluator::new(&setup.ntt, &setup.encoder, Some(&setup.keys.eval_key));
    
    let mut harvester = ShadowHarvester::with_seed(42);
    let ct = encryptor.encrypt(17, &mut harvester);
    
    group.bench_function("light_N1024", |b| {
        b.iter(|| {
            evaluator.mul_plain(black_box(&ct), black_box(5))
        });
    });
    
    group.finish();
}

// =============================================================================
// NTT BENCHMARKS
// =============================================================================

fn bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    
    let q = 998244353u64;
    
    for n in [512, 1024, 2048, 4096] {
        let ntt = NTTEngine::new(q, n);
        let data: Vec<u64> = (0..n as u64).collect();
        
        group.bench_function(
            BenchmarkId::new("forward", n),
            |b| {
                b.iter(|| ntt.ntt(black_box(&data)));
            },
        );
        
        let ntt_data = ntt.ntt(&data);
        group.bench_function(
            BenchmarkId::new("inverse", n),
            |b| {
                b.iter(|| ntt.intt(black_box(&ntt_data)));
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// ENTROPY BENCHMARKS
// =============================================================================

fn bench_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy");
    
    // Shadow Entropy
    group.bench_function("shadow_u64", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| harvester.next_u64());
    });
    
    // OS CSPRNG
    group.bench_function("secure_u64", |b| {
        b.iter(|| qmnf_fhe::entropy::secure_u64());
    });
    
    group.bench_function("shadow_ternary_1024", |b| {
        let mut harvester = ShadowHarvester::with_seed(42);
        b.iter(|| {
            for _ in 0..1024 {
                black_box(harvester.ternary());
            }
        });
    });
    
    group.bench_function("secure_ternary_1024", |b| {
        b.iter(|| qmnf_fhe::entropy::secure_ternary_vector(1024));
    });
    
    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

criterion_group!(
    benches,
    bench_keygen,
    bench_encrypt,
    bench_decrypt,
    bench_homo_add,
    bench_homo_mul,
    bench_mul_plain,
    bench_ntt,
    bench_entropy,
);

criterion_main!(benches);
