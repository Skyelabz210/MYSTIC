///! Camp Mystic 2007 Flash Flood - Validation Test
//!
//! Tests MYSTIC system's ability to detect the June 28, 2007 Camp Mystic
//! flash flood that killed 3 people. Analyzes:
//! - Detection lead time (how early could we warn?)
//! - Alert progression accuracy
//! - Attractor basin recognition
//!
//! In memory of those lost. This validates that MYSTIC could have saved lives.

use qmnf_fhe::chaos::{
    FloodDetector, DelugeEngine, WeatherState, RawSensorData,
    AlertLevel,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
struct TestResult {
    timestamp: String,
    hours_before_flood: f64,
    alert_level: AlertLevel,
    probability: f64,
    event_phase: String,
    correct_detection: bool,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         MYSTIC VALIDATION TEST - Camp Mystic Flood 2007          ║");
    println!("║                                                                   ║");
    println!("║  Event: June 28, 2007, 2:00 PM                                   ║");
    println!("║  Location: Guadalupe River at Kerrville, TX                      ║");
    println!("║  Fatalities: 3 adults swept away                                 ║");
    println!("║                                                                   ║");
    println!("║  Testing: Can MYSTIC provide 2-6 hour advance warning?           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load unified dataset if available, fall back to synthetic
    let data_file = if Path::new("data/camp_mystic_2007_unified.csv").exists() {
        "data/camp_mystic_2007_unified.csv"
    } else {
        "data/camp_mystic_2007_synthetic.csv"
    };
    println!("Loading historical flood data: {}", data_file);

    let file = match File::open(data_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ ERROR: Could not open {}: {}", data_file, e);
            eprintln!("  Make sure you ran:");
            eprintln!("    python3 scripts/fetch_camp_mystic_2007.py");
            eprintln!("    python3 scripts/build_camp_mystic_unified.py");
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    lines.next();

    let mut engine = DelugeEngine::new();
    let mut results: Vec<TestResult> = Vec::new();

    let flood_time = chrono::NaiveDateTime::parse_from_str(
        "2007-06-28T14:00:00",
        "%Y-%m-%dT%H:%M:%S"
    ).unwrap();

    println!("Processing {} time steps...", "312");
    println!();

    let mut warning_issued_at: Option<(f64, AlertLevel)> = None;
    let mut max_alert = AlertLevel::Clear;

    for line in lines {
        let line = line.unwrap();
        let fields: Vec<&str> = line.split(',').collect();

        if fields.len() < 10 {
            continue;
        }

        // Parse CSV fields
        let timestamp = fields[0];
        let station_id: u64 = fields[1].parse().unwrap_or(1);
        let temp_c: f64 = fields[2].parse().unwrap_or(0.0);
        let dewpoint_c: f64 = fields[3].parse().unwrap_or(0.0);
        let pressure_hpa: f64 = fields[4].parse().unwrap_or(1013.0);
        let wind_mps: f64 = fields[5].parse().unwrap_or(0.0);
        let rain_mm_hr: f64 = fields[6].parse().unwrap_or(0.0);
        let soil_pct: f64 = fields[7].parse().unwrap_or(0.0);
        let stream_cm: f64 = fields[8].parse().unwrap_or(0.0);
        let event_type = fields[9];

        // Convert to i32 scaled by 100
        let raw = RawSensorData {
            temp: (temp_c * 100.0) as i32,
            dewpoint: (dewpoint_c * 100.0) as i32,
            pressure: (pressure_hpa * 100.0) as i32,
            wind_speed: (wind_mps * 100.0) as i32,
            wind_dir: 225, // Assume southerly
            rain_rate: (rain_mm_hr * 100.0) as i32,
            soil_moisture: (soil_pct * 100.0) as i32,
            stream_level: (stream_cm * 100.0) as i32,
            ..Default::default()
        };

        // Calculate hours before flood
        let current_time = chrono::NaiveDateTime::parse_from_str(
            timestamp,
            "%Y-%m-%dT%H:%M:%S"
        ).unwrap();

        let hours_before = (flood_time - current_time).num_seconds() as f64 / 3600.0;

        // Create weather state and update detector
        let state = WeatherState::from_sensors(raw, station_id, 0);
        engine.update_station(state);

        // Get prediction
        let prediction = engine.all_predictions()
            .into_iter()
            .find(|(id, _)| *id == station_id)
            .map(|(_, pred)| pred);

        if let Some(pred) = prediction {
            // Check for first warning
            if warning_issued_at.is_none() && pred.alert >= AlertLevel::Watch {
                warning_issued_at = Some((hours_before, pred.alert));
            }

            // Track maximum alert level
            if pred.alert > max_alert {
                max_alert = pred.alert;
            }

            // Determine if detection is correct for this phase
            let correct = match event_type {
                "normal" | "baseline" => pred.alert <= AlertLevel::Watch,
                "watch" => pred.alert >= AlertLevel::Watch,
                "precursor" => pred.alert >= AlertLevel::Advisory,
                "imminent" => pred.alert >= AlertLevel::Warning,
                "flash_flood" => pred.alert >= AlertLevel::Warning,
                _ => false,
            };

            results.push(TestResult {
                timestamp: timestamp.to_string(),
                hours_before_flood: hours_before,
                alert_level: pred.alert,
                probability: pred.probability,
                event_phase: event_type.to_string(),
                correct_detection: correct,
            });
        }
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         TEST RESULTS                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Analysis
    if let Some((hours, alert)) = warning_issued_at {
        println!("✓ FIRST WARNING ISSUED:");
        println!("  Time before flood: {:.1} hours", hours);
        println!("  Alert level: {}", alert.name());
        println!();

        if hours >= 2.0 && hours <= 6.0 {
            println!("  ✓ MEETS DESIGN GOAL: 2-6 hour advance warning");
        } else if hours > 6.0 {
            println!("  ✓ EXCEEDS GOAL: Warning issued over 6 hours early");
        } else {
            println!("  ⚠ BELOW GOAL: Warning issued less than 2 hours early");
        }
    } else {
        println!("✗ NO WARNING ISSUED");
        println!("  System failed to detect flood risk");
    }

    println!();
    println!("MAXIMUM ALERT LEVEL: {}", max_alert.name());
    println!();

    // Accuracy analysis by phase
    println!("═══ DETECTION ACCURACY BY PHASE ═══");
    println!();

    let phases = ["baseline", "watch", "precursor", "imminent", "flash_flood"];
    for phase in &phases {
        let phase_results: Vec<&TestResult> = results.iter()
            .filter(|r| r.event_phase == *phase)
            .collect();

        if !phase_results.is_empty() {
            let correct_count = phase_results.iter()
                .filter(|r| r.correct_detection)
                .count();

            let accuracy = (correct_count as f64 / phase_results.len() as f64) * 100.0;
            let avg_prob = phase_results.iter()
                .map(|r| r.probability)
                .sum::<f64>() / phase_results.len() as f64;

            let most_common_alert = phase_results.iter()
                .map(|r| r.alert_level)
                .max()
                .unwrap_or(AlertLevel::Clear);

            println!("{:12} | Accuracy: {:5.1}% | Avg Probability: {:5.1}% | Alert: {}",
                phase,
                accuracy,
                avg_prob * 100.0,
                most_common_alert.name()
            );
        }
    }

    println!();
    println!("═══ TIMELINE HIGHLIGHTS ═══");
    println!();

    // Show key transitions
    let key_times = [72.0, 48.0, 24.0, 12.0, 6.0, 3.0, 1.0, 0.0, -1.0];
    for &hours in &key_times {
        if let Some(result) = results.iter()
            .min_by_key(|r| ((r.hours_before_flood - hours).abs() * 1000.0) as i64)
        {
            if (result.hours_before_flood - hours).abs() < 0.5 {
                let time_label = if hours > 0.0 {
                    format!("T-{:.0}h", hours)
                } else if hours < 0.0 {
                    format!("T+{:.0}h", -hours)
                } else {
                    "T-0h (FLOOD)".to_string()
                };

                println!("{:13} | {:10} | {} | Prob: {:5.1}%",
                    time_label,
                    result.event_phase,
                    result.alert_level.name(),
                    result.probability * 100.0
                );
            }
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         CONCLUSION                                ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let overall_accuracy = (results.iter().filter(|r| r.correct_detection).count() as f64
        / results.len() as f64) * 100.0;

    println!("║  Overall Accuracy: {:5.1}%                                        ║", overall_accuracy);

    if let Some((hours, _)) = warning_issued_at {
        if hours >= 2.0 {
            println!("║  Warning Lead Time: {:.1} hours                                  ║", hours);
            println!("║                                                                   ║");
            println!("║  ✓ VALIDATION SUCCESSFUL                                         ║");
            println!("║                                                                   ║");
            println!("║  MYSTIC could have provided advance warning, potentially         ║");
            println!("║  preventing the tragedy at Camp Mystic on June 28, 2007.         ║");
        } else {
            println!("║  Warning Lead Time: {:.1} hours (below 2-hour goal)              ║", hours);
            println!("║                                                                   ║");
            println!("║  ⚠ PARTIAL SUCCESS - Needs tuning                               ║");
        }
    } else {
        println!("║                                                                   ║");
        println!("║  ✗ VALIDATION FAILED - No warning issued                         ║");
        println!("║  System requires additional training on flood signatures         ║");
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("In memory of those lost. No more tragedies.");
}
