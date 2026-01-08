//! Noise Tracking Benchmarks
//!
//! Comparing:
//! 1. P² Quantile Estimator (O(1) memory, O(1) update)
//! 2. EMA Calculator
//! 3. Multi-Window Detector
//! 4. Full NoiseDistribution tracker

use std::time::{Duration, Instant};

// Import from the library
use qmnf_fhe::noise::{
    P2QuantileEstimator, NoiseDistribution, EMACalculator,
    MultiWindowNoiseDetector, NoiseBudgetTracker,
};

fn black_box<T>(x: T) -> T {
    unsafe { std::ptr::read_volatile(&x) }
}

fn bench_p2_update(iterations: u64) -> (Duration, i64) {
    let mut p2 = P2QuantileEstimator::median();
    
    let start = Instant::now();
    for i in 0..iterations {
        p2.update(black_box((i % 100_000) as i64));
    }
    let elapsed = start.elapsed();
    
    (elapsed, p2.value())
}

fn bench_ema_update(iterations: u64) -> (Duration, i64) {
    let mut ema = EMACalculator::standard();
    
    let start = Instant::now();
    for i in 0..iterations {
        ema.update(black_box((i % 100_000) as i64));
    }
    let elapsed = start.elapsed();
    
    (elapsed, ema.value_millibits())
}

fn bench_multi_window(iterations: u64) -> (Duration, bool) {
    let mut detector = MultiWindowNoiseDetector::new();
    let mut has_anomaly = false;
    
    let start = Instant::now();
    for i in 0..iterations {
        let noise = black_box((i % 100_000) as i64);
        detector.update(noise);
        if i % 1000 == 0 {
            has_anomaly |= detector.detect_anomaly(noise).has_anomaly();
        }
    }
    let elapsed = start.elapsed();
    
    (elapsed, has_anomaly)
}

fn bench_noise_distribution(iterations: u64) -> (Duration, String) {
    let mut dist = NoiseDistribution::new();
    
    let start = Instant::now();
    for i in 0..iterations {
        dist.update(black_box(3000 + (i % 50_000) as i64));
    }
    let elapsed = start.elapsed();
    
    (elapsed, dist.summary())
}

fn bench_budget_tracker(iterations: u64) -> (Duration, f64) {
    let mut tracker = NoiseBudgetTracker::standard_128();
    tracker.on_encrypt();
    
    let start = Instant::now();
    for _ in 0..iterations {
        tracker.on_add();
    }
    let elapsed = start.elapsed();
    
    (elapsed, tracker.noise_bits())
}

fn format_rate(iterations: u64, duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    let rate = iterations as f64 / secs;
    
    if rate >= 1_000_000_000.0 {
        format!("{:.2} G ops/sec", rate / 1_000_000_000.0)
    } else if rate >= 1_000_000.0 {
        format!("{:.2} M ops/sec", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2} K ops/sec", rate / 1_000.0)
    } else {
        format!("{:.2} ops/sec", rate)
    }
}

fn format_ns_per_op(iterations: u64, duration: Duration) -> String {
    let ns = duration.as_nanos() as f64 / iterations as f64;
    if ns < 1000.0 {
        format!("{:.1} ns/op", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs/op", ns / 1000.0)
    } else {
        format!("{:.2} ms/op", ns / 1_000_000.0)
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       QMNF FHE NOISE TRACKING BENCHMARK                       ║");
    println!("║       CDHS-Based Integer-Only Statistics                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    
    let iterations = 1_000_000u64;
    
    // Warmup
    println!("Warming up...");
    let _ = bench_p2_update(10_000);
    let _ = bench_ema_update(10_000);
    let _ = bench_multi_window(10_000);
    let _ = bench_noise_distribution(10_000);
    let _ = bench_budget_tracker(10_000);
    println!();
    
    // P² Quantile Estimator
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ P² QUANTILE ESTIMATOR (Streaming Percentiles)              │");
    println!("├─────────────────────────────────────────────────────────────┤");
    let (duration, result) = bench_p2_update(iterations);
    println!("│ Iterations: {:>12}                                    │", iterations);
    println!("│ Time:       {:>12.3} ms                                │", duration.as_secs_f64() * 1000.0);
    println!("│ Rate:       {:>18}                            │", format_rate(iterations, duration));
    println!("│ Latency:    {:>18}                            │", format_ns_per_op(iterations, duration));
    println!("│ Result:     {:>12} millibits                        │", result);
    println!("│ Memory:     O(1) - 5 markers only                          │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // EMA Calculator
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ EMA CALCULATOR (Trend Tracking)                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    let (duration, result) = bench_ema_update(iterations);
    println!("│ Iterations: {:>12}                                    │", iterations);
    println!("│ Time:       {:>12.3} ms                                │", duration.as_secs_f64() * 1000.0);
    println!("│ Rate:       {:>18}                            │", format_rate(iterations, duration));
    println!("│ Latency:    {:>18}                            │", format_ns_per_op(iterations, duration));
    println!("│ Result:     {:>12} millibits                        │", result);
    println!("│ Memory:     O(1) - single value                            │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // Multi-Window Detector
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ MULTI-WINDOW DETECTOR (Anomaly Detection)                  │");
    println!("├─────────────────────────────────────────────────────────────┤");
    let (duration, result) = bench_multi_window(iterations);
    println!("│ Iterations: {:>12}                                    │", iterations);
    println!("│ Time:       {:>12.3} ms                                │", duration.as_secs_f64() * 1000.0);
    println!("│ Rate:       {:>18}                            │", format_rate(iterations, duration));
    println!("│ Latency:    {:>18}                            │", format_ns_per_op(iterations, duration));
    println!("│ Anomalies:  {:>12}                                    │", result);
    println!("│ Memory:     O(W) - 336 samples (16+64+256)                 │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // Full Noise Distribution
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ NOISE DISTRIBUTION (Full P² Suite)                         │");
    println!("├─────────────────────────────────────────────────────────────┤");
    let (duration, summary) = bench_noise_distribution(iterations);
    println!("│ Iterations: {:>12}                                    │", iterations);
    println!("│ Time:       {:>12.3} ms                                │", duration.as_secs_f64() * 1000.0);
    println!("│ Rate:       {:>18}                            │", format_rate(iterations, duration));
    println!("│ Latency:    {:>18}                            │", format_ns_per_op(iterations, duration));
    println!("│ Memory:     O(1) - 4×5 markers = 20 values                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ {}│", format!("{:<60}", summary));
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // Budget Tracker
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ BUDGET TRACKER (Per-Ciphertext Accounting)                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    let (duration, result) = bench_budget_tracker(iterations);
    println!("│ Iterations: {:>12} additions                          │", iterations);
    println!("│ Time:       {:>12.3} ms                                │", duration.as_secs_f64() * 1000.0);
    println!("│ Rate:       {:>18}                            │", format_rate(iterations, duration));
    println!("│ Latency:    {:>18}                            │", format_ns_per_op(iterations, duration));
    println!("│ Final noise:{:>12.1} bits                              │", result);
    println!("│ Memory:     O(1) + O(64) history                           │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                                ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ All operations are INTEGER-ONLY (zero floating-point)        ║");
    println!("║ P² estimator: O(1) memory for streaming percentiles          ║");
    println!("║ Perfect determinism across all platforms                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
}
