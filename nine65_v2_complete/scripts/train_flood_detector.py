#!/usr/bin/env python3
"""
MYSTIC Training Script - Flash Flood Attractor Learning

Reads historical weather/stream data and trains the FloodDetector
to recognize attractor basin signatures associated with flash floods.

Uses the NINE65 exact arithmetic engine via Rust FFI.
"""

import csv
import sys
from datetime import datetime
from typing import List, Dict


def load_training_data(csv_file: str) -> List[Dict]:
    """
    Load MYSTIC CSV training data.

    Returns:
        List of dictionaries with sensor readings
    """
    data = []

    print(f"Loading training data from: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            record = {
                'timestamp': row['timestamp'],
                'station_id': int(row['station_id']),
                'temp_c': float(row['temp_c']),
                'dewpoint_c': float(row['dewpoint_c']),
                'pressure_hpa': float(row['pressure_hpa']),
                'wind_mps': float(row['wind_mps']),
                'rain_mm_hr': float(row['rain_mm_hr']),
                'soil_pct': float(row['soil_pct']),
                'stream_cm': float(row['stream_cm']),
                'event_type': row['event_type']
            }
            data.append(record)

    print(f"Loaded {len(data)} records")
    return data


def analyze_data_quality(data: List[Dict]):
    """
    Print data quality statistics.
    """
    print("\n" + "─" * 60)
    print("DATA QUALITY ANALYSIS")
    print("─" * 60)

    # Count by station
    stations = {}
    for record in data:
        sid = record['station_id']
        stations[sid] = stations.get(sid, 0) + 1

    print(f"\nRecords by station:")
    for sid, count in sorted(stations.items()):
        print(f"  Station {sid}: {count:,} records")

    # Count by event type
    events = {}
    for record in data:
        et = record['event_type']
        events[et] = events.get(et, 0) + 1

    print(f"\nRecords by event type:")
    for et, count in sorted(events.items()):
        print(f"  {et}: {count:,} records ({100*count/len(data):.1f}%)")

    # Stream level statistics
    stream_levels = [r['stream_cm'] for r in data if r['stream_cm'] > 0]
    if stream_levels:
        print(f"\nStream level statistics (cm):")
        print(f"  Min:    {min(stream_levels):.2f}")
        print(f"  Max:    {max(stream_levels):.2f}")
        print(f"  Mean:   {sum(stream_levels)/len(stream_levels):.2f}")
        print(f"  Median: {sorted(stream_levels)[len(stream_levels)//2]:.2f}")


def generate_rust_training_code(data: List[Dict], output_file: str):
    """
    Generate Rust code that uses the training data.

    Since we're calling Rust from Python, we'll generate a Rust binary
    that loads this data and trains the FloodDetector.
    """
    # Filter to only flood events for training
    flood_data = [r for r in data if r['event_type'] in ['flash_flood', 'major_flood']]
    normal_data = [r for r in data if r['event_type'] == 'normal']

    print("\n" + "─" * 60)
    print("GENERATING RUST TRAINING BINARY")
    print("─" * 60)
    print(f"Flood events: {len(flood_data)}")
    print(f"Normal events: {len(normal_data)} (will sample)")

    # Sample normal data to balance dataset
    sample_size = min(len(normal_data), len(flood_data) * 10)
    # QMNF: Use deterministic sampling instead of random.sample
    # Sample evenly spaced elements for reproducibility
    if normal_data and sample_size > 0:
        step = max(1, len(normal_data) // sample_size)
        normal_sample = [normal_data[i] for i in range(0, len(normal_data), step)][:sample_size]
    else:
        normal_sample = []

    rust_code = '''//! MYSTIC Training Binary - Auto-generated
//!
//! Trains FloodDetector on historical Texas Hill Country data.

use qmnf_fhe::chaos::{FloodDetector, WeatherState, RawSensorData};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         MYSTIC Flash Flood Attractor Training            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let mut detector = FloodDetector::new();

    // Training data: Texas Hill Country historical floods
    println!("Loading training data...");
    println!();

    // Flood event data
'''

    # Add flood events
    rust_code += f"    let flood_events: Vec<WeatherState> = vec![\n"
    for i, record in enumerate(flood_data[:100]):  # Limit to 100 for code size
        temp = int(record['temp_c'] * 100)
        dewpoint = int(record['dewpoint_c'] * 100)
        pressure = int(record['pressure_hpa'] * 100)
        wind = int(record['wind_mps'] * 100)
        rain = int(record['rain_mm_hr'] * 100)
        soil = int(record['soil_pct'] * 100)
        stream = int(record['stream_cm'] * 100)
        station_id = record['station_id']

        # Parse timestamp to unix seconds (simplified)
        ts = 1700000000 + i * 600  # Just use incrementing timestamps

        rust_code += f'''        WeatherState::from_sensors(
            RawSensorData {{
                temp: {temp},
                dewpoint: {dewpoint},
                pressure: {pressure},
                wind_speed: {wind},
                wind_dir: 225,
                rain_rate: {rain},
                soil_moisture: {soil},
                stream_level: {stream},
                ..Default::default()
            }},
            {station_id},
            {ts}
        ),
'''

    rust_code += '''    ];

    println!("Training on {} flood events...", flood_events.len());
    detector.learn_flood_event(&flood_events);
    println!("✓ Training complete");
    println!();

    // Test on a sample event
    if let Some(test_event) = flood_events.first() {
        detector.update(test_event);
        let prediction = detector.predict();

        println!("Test prediction on known flood event:");
        println!("  Probability: {:.1}%", prediction.probability * 100.0);
        println!("  Alert: {}", prediction.alert.name());
        println!("  Action: {}", prediction.action);
        println!("  Lyapunov: {:.4}", detector.lyapunov());
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                  TRAINING COMPLETE                         ║");
    println!("║  Detector ready for deployment                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}
'''

    with open(output_file, 'w') as f:
        f.write(rust_code)

    print(f"\nWrote Rust training binary to: {output_file}")
    print("\nTo compile and run:")
    print(f"  cd /home/acid/Downloads/nine65_v2_complete")
    print(f"  cargo run --release --bin train_mystic --features v2")


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          MYSTIC Training - Flood Attractor Learning       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Load data
    data_file = "../data/texas_hill_country_usgs.csv"
    data = load_training_data(data_file)

    # Analyze quality
    analyze_data_quality(data)

    # Generate Rust training binary
    output_file = "../src/bin/train_mystic.rs"
    generate_rust_training_code(data, output_file)

    print("\n" + "═" * 60)
    print("Next steps:")
    print("  1. Compile training binary:")
    print("     cargo build --release --bin train_mystic --features v2")
    print("  2. Run training:")
    print("     cargo run --release --bin train_mystic --features v2")
    print("  3. Test with live data")
    print("═" * 60)


if __name__ == "__main__":
    main()
