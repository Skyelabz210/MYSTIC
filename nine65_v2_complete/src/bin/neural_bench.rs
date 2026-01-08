//! Neural FHE Benchmarks
//!
//! Benchmarks for QMNF FHE neural network innovations

use qmnf_fhe::prelude::*;
use qmnf_fhe::arithmetic::{PADE_SCALE, SOFTMAX_SCALE};
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                    QMNF FHE NEURAL BENCHMARKS                                 ");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    benchmark_pade();
    benchmark_mq_relu();
    benchmark_mobius();
    benchmark_softmax();
    benchmark_neural_layer();
}

fn benchmark_pade() {
    println!("PADÉ [4/4] ENGINE");
    println!("─────────────────────────────────────────────────────────────────────────────");
    
    let pade = PadeEngine::default();
    let iterations = 1_000_000;
    
    // exp()
    let start = Instant::now();
    let mut sum = 0i128;
    for i in 0..iterations {
        sum += pade.exp_integer((i % 1000) as i128 * PADE_SCALE / 1000);
    }
    let elapsed = start.elapsed();
    println!("  exp():     {:>8.1}ns/op  (sum={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, sum % 1000);
    
    // sigmoid()
    let start = Instant::now();
    let mut sum = 0i128;
    for i in 0..iterations {
        sum += pade.sigmoid_integer((i % 2000) as i128 * PADE_SCALE / 1000 - PADE_SCALE);
    }
    let elapsed = start.elapsed();
    println!("  sigmoid(): {:>8.1}ns/op  (sum={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, sum % 1000);
    
    // tanh()
    let start = Instant::now();
    let mut sum = 0i128;
    for i in 0..iterations {
        sum += pade.tanh_integer((i % 2000) as i128 * PADE_SCALE / 1000 - PADE_SCALE);
    }
    let elapsed = start.elapsed();
    println!("  tanh():    {:>8.1}ns/op  (sum={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, sum % 1000);
    
    println!();
}

fn benchmark_mq_relu() {
    println!("MQ-RELU O(1) SIGN DETECTION");
    println!("─────────────────────────────────────────────────────────────────────────────");
    
    let relu = MQReLU::new(998244353);
    let iterations = 10_000_000;
    
    // Single scalar ReLU
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(relu.apply_scalar((i as u64 * 12345) % 998244353));
    }
    let elapsed = start.elapsed();
    println!("  scalar:    {:>8.1}ns/op  (sum={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, sum % 1000);
    
    // Polynomial ReLU (1024 coefficients)
    let poly: Vec<u64> = (0..1024).map(|i| (i * 97531) % 998244353).collect();
    let poly_iterations = 100_000;
    
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..poly_iterations {
        let result = relu.apply_polynomial(&poly);
        sum = sum.wrapping_add(result[0]);
    }
    let elapsed = start.elapsed();
    println!("  poly(1024): {:>7.1}µs/op  (sum={} to prevent optimization)", 
             elapsed.as_micros() as f64 / poly_iterations as f64, sum % 1000);
    
    println!();
}

fn benchmark_mobius() {
    println!("MOBIUSINT SIGNED ARITHMETIC");
    println!("─────────────────────────────────────────────────────────────────────────────");
    
    let iterations = 10_000_000;
    
    // Addition
    let a = MobiusInt::from_i64(12345);
    let b = MobiusInt::from_i64(-6789);
    
    let start = Instant::now();
    let mut result = MobiusInt::zero();
    for _ in 0..iterations {
        result = result.add(&a).add(&b);
    }
    let elapsed = start.elapsed();
    println!("  add():     {:>8.1}ns/op  (result={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, result.spinor_value() % 1000);
    
    // Multiplication
    let start = Instant::now();
    let mut result = MobiusInt::from_i64(1);
    for i in 0..iterations {
        let x = MobiusInt::from_i64((i % 100) as i64 - 50);
        result = result.add(&a.mul(&x));
    }
    let elapsed = start.elapsed();
    println!("  mul():     {:>8.1}ns/op  (result={} to prevent optimization)", 
             elapsed.as_nanos() as f64 / iterations as f64, result.spinor_value() % 1000);
    
    println!();
}

fn benchmark_softmax() {
    println!("INTEGER SOFTMAX (EXACT SUM)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    
    let softmax = IntegerSoftmax::new();
    
    // 10-class softmax
    let logits_10: Vec<i128> = (0..10).map(|i| (i * 100_000) as i128).collect();
    let iterations_10 = 100_000;
    
    let start = Instant::now();
    let mut sum = 0u128;
    for _ in 0..iterations_10 {
        let result = softmax.compute(&logits_10);
        sum = sum.wrapping_add(result[0]);
    }
    let elapsed = start.elapsed();
    let probs = softmax.compute(&logits_10);
    let total: u128 = probs.iter().sum();
    println!("  10-class:  {:>8.1}µs/op  (sum={}, exact_sum={})", 
             elapsed.as_micros() as f64 / iterations_10 as f64, 
             sum % 1000,
             total == SOFTMAX_SCALE);
    
    // 100-class softmax
    let logits_100: Vec<i128> = (0..100).map(|i| (i * 10_000) as i128).collect();
    let iterations_100 = 10_000;
    
    let start = Instant::now();
    let mut sum = 0u128;
    for _ in 0..iterations_100 {
        let result = softmax.compute(&logits_100);
        sum = sum.wrapping_add(result[0]);
    }
    let elapsed = start.elapsed();
    let probs = softmax.compute(&logits_100);
    let total: u128 = probs.iter().sum();
    println!("  100-class: {:>8.1}µs/op  (sum={}, exact_sum={})", 
             elapsed.as_micros() as f64 / iterations_100 as f64, 
             sum % 1000,
             total == SOFTMAX_SCALE);
    
    // 1000-class softmax  
    let logits_1000: Vec<i128> = (0..1000).map(|i| (i * 1_000) as i128).collect();
    let iterations_1000 = 1_000;
    
    let start = Instant::now();
    let mut sum = 0u128;
    for _ in 0..iterations_1000 {
        let result = softmax.compute(&logits_1000);
        sum = sum.wrapping_add(result[0]);
    }
    let elapsed = start.elapsed();
    let probs = softmax.compute(&logits_1000);
    let total: u128 = probs.iter().sum();
    println!("  1000-class:{:>8.1}µs/op  (sum={}, exact_sum={})", 
             elapsed.as_micros() as f64 / iterations_1000 as f64, 
             sum % 1000,
             total == SOFTMAX_SCALE);
    
    println!();
}

fn benchmark_neural_layer() {
    println!("FHE NEURAL EVALUATOR");
    println!("─────────────────────────────────────────────────────────────────────────────");
    
    let eval = FHENeuralEvaluator::new(998244353, 65537);
    
    // Dense layer 64→64 with ReLU
    let input: Vec<MobiusInt> = (0..64)
        .map(|i| MobiusInt::from_i64((i * 100 - 3200) as i64))
        .collect();
    
    let weights: Vec<Vec<MobiusInt>> = (0..64)
        .map(|i| {
            (0..64)
                .map(|j| MobiusInt::from_i64(((i * 64 + j) % 200 - 100) as i64))
                .collect()
        })
        .collect();
    
    let biases: Vec<MobiusInt> = (0..64)
        .map(|i| MobiusInt::from_i64((i * 10 - 320) as i64))
        .collect();
    
    let iterations = 10_000;
    
    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..iterations {
        let output = eval.dense_forward(&input, &weights, &biases, ActivationType::ReLU);
        sum = sum.wrapping_add(output[0].spinor_value());
    }
    let elapsed = start.elapsed();
    println!("  dense(64→64,ReLU): {:>8.1}µs/op  (sum={} to prevent optimization)", 
             elapsed.as_micros() as f64 / iterations as f64, sum % 1000);
    
    // Dense layer 256→256 with Sigmoid
    let input_256: Vec<MobiusInt> = (0..256)
        .map(|i| MobiusInt::from_i64((i * 50 - 6400) as i64))
        .collect();
    
    let weights_256: Vec<Vec<MobiusInt>> = (0..256)
        .map(|i| {
            (0..256)
                .map(|j| MobiusInt::from_i64(((i * 256 + j) % 100 - 50) as i64))
                .collect()
        })
        .collect();
    
    let biases_256: Vec<MobiusInt> = (0..256)
        .map(|i| MobiusInt::from_i64((i * 5 - 640) as i64))
        .collect();
    
    let iterations_256 = 1_000;
    
    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..iterations_256 {
        let output = eval.dense_forward(&input_256, &weights_256, &biases_256, ActivationType::None);
        sum = sum.wrapping_add(output[0].spinor_value());
    }
    let elapsed = start.elapsed();
    println!("  dense(256→256,None): {:>6.1}ms/op  (sum={} to prevent optimization)", 
             elapsed.as_millis() as f64 / iterations_256 as f64, sum % 1000);
    
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                         BENCHMARK COMPLETE                                    ");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}
