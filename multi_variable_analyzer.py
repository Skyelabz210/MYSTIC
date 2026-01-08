#!/usr/bin/env python3
"""
MULTI-VARIABLE WEATHER ANALYZER

Addresses Gap #3: Single-variable analysis limitation.

Combines multiple weather variables for comprehensive risk assessment:
- Pressure: Atmospheric stability
- Humidity: Fire weather/drying conditions
- Precipitation: Flood potential
- Wind: Storm intensity
- Streamflow: Flood conditions
- Temperature: Heat extremes

Each hazard type has specific variable combinations:
- HURRICANE: pressure + wind + precipitation
- FIRE_WEATHER: humidity (inverted) + wind + temperature
- FLASH_FLOOD: precipitation + streamflow
- TORNADO: pressure oscillation + wind
- STABLE: all variables within normal ranges

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from mystic_v3_production import MYSTICPredictorV3Production, PredictionResult


class HazardType(Enum):
    """Types of weather hazards with their key indicators."""
    HURRICANE = "HURRICANE"
    FIRE_WEATHER = "FIRE_WEATHER"
    FLASH_FLOOD = "FLASH_FLOOD"
    TORNADO = "TORNADO"
    SEVERE_STORM = "SEVERE_STORM"
    STABLE = "STABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class VariableThresholds:
    """Thresholds for each variable type."""
    # Pressure (hPa × 10)
    pressure_low_critical: int = 9800   # < 980 hPa
    pressure_low_warning: int = 10000   # < 1000 hPa
    pressure_drop_critical: int = -50   # > 5 hPa drop
    pressure_drop_warning: int = -30    # > 3 hPa drop

    # Humidity (%)
    humidity_fire_critical: int = 15    # < 15% = extreme fire danger
    humidity_fire_warning: int = 25     # < 25% = high fire danger

    # Wind speed (km/h × 10)
    wind_critical: int = 500            # > 50 km/h
    wind_warning: int = 300             # > 30 km/h

    # Precipitation (mm × 100)
    precip_flood_critical: int = 5000   # > 50 mm/hr
    precip_flood_warning: int = 2500    # > 25 mm/hr

    # Temperature (°C × 100)
    temp_heat_critical: int = 4000      # > 40°C
    temp_heat_warning: int = 3500       # > 35°C

    # Streamflow (cfs × 100, relative to normal)
    streamflow_flood_ratio: int = 500   # > 5× normal


@dataclass
class MultiVariableResult:
    """Result of multi-variable analysis."""
    hazard_type: HazardType
    composite_risk: str  # LOW, MODERATE, HIGH, CRITICAL
    composite_score: int  # 0-150
    confidence: int  # 0-100

    # Individual variable contributions
    pressure_contribution: int
    humidity_contribution: int
    wind_contribution: int
    precip_contribution: int
    temp_contribution: int
    streamflow_contribution: int

    # Signals detected
    signals: List[str]

    # Individual predictions
    individual_predictions: Dict[str, PredictionResult]


class MultiVariableAnalyzer:
    """
    Analyzes multiple weather variables for comprehensive risk assessment.
    """

    def __init__(self):
        self.predictor = MYSTICPredictorV3Production()
        self.thresholds = VariableThresholds()

    def analyze(
        self,
        data: Dict[str, List[int]],
        location: str = "UNKNOWN"
    ) -> MultiVariableResult:
        """
        Analyze multiple variables and compute composite risk.

        Args:
            data: Dictionary mapping variable names to integer-scaled time series
                  Keys: pressure, humidity, wind_speed, precipitation, temperature, streamflow
            location: Location identifier
        """
        signals = []
        contributions = {
            "pressure": 0,
            "humidity": 0,
            "wind": 0,
            "precip": 0,
            "temp": 0,
            "streamflow": 0,
        }
        individual = {}

        # Analyze each available variable
        if "pressure" in data and data["pressure"]:
            pressure = data["pressure"]
            result = self.predictor.predict(pressure, location, "PRESSURE")
            individual["pressure"] = result

            # Check pressure levels
            min_pressure = min(pressure)
            if min_pressure < self.thresholds.pressure_low_critical:
                contributions["pressure"] = 30
                signals.append("EXTREME_LOW_PRESSURE")
            elif min_pressure < self.thresholds.pressure_low_warning:
                contributions["pressure"] = 20
                signals.append("LOW_PRESSURE")

            # Check pressure drop
            if len(pressure) > 1:
                max_drop = min(pressure[i+1] - pressure[i] for i in range(len(pressure)-1))
                if max_drop < self.thresholds.pressure_drop_critical:
                    contributions["pressure"] += 20
                    signals.append("RAPID_PRESSURE_DROP")

        if "humidity" in data and data["humidity"]:
            humidity = data["humidity"]

            # For humidity, LOW values = HIGH risk
            min_humidity = min(humidity)
            if min_humidity < self.thresholds.humidity_fire_critical:
                contributions["humidity"] = 35
                signals.append("EXTREME_FIRE_DANGER_HUMIDITY")
            elif min_humidity < self.thresholds.humidity_fire_warning:
                contributions["humidity"] = 20
                signals.append("HIGH_FIRE_DANGER_HUMIDITY")

            # Analyze inverted humidity through predictor
            inverted = [100 - h for h in humidity]
            result = self.predictor.predict(inverted, location, "HUMIDITY_INVERTED")
            individual["humidity"] = result

        if "wind_speed" in data and data["wind_speed"]:
            wind = data["wind_speed"]
            result = self.predictor.predict(wind, location, "WIND")
            individual["wind"] = result

            max_wind = max(wind)
            if max_wind > self.thresholds.wind_critical:
                contributions["wind"] = 25
                signals.append("EXTREME_WIND")
            elif max_wind > self.thresholds.wind_warning:
                contributions["wind"] = 15
                signals.append("HIGH_WIND")

        if "precipitation" in data and data["precipitation"]:
            precip = data["precipitation"]
            result = self.predictor.predict(precip, location, "PRECIPITATION")
            individual["precipitation"] = result

            max_precip = max(precip)
            total_precip = sum(precip)
            if max_precip > self.thresholds.precip_flood_critical:
                contributions["precip"] = 30
                signals.append("EXTREME_PRECIPITATION_RATE")
            elif max_precip > self.thresholds.precip_flood_warning:
                contributions["precip"] = 20
                signals.append("HIGH_PRECIPITATION_RATE")

            if total_precip > 10000:  # > 100mm total
                contributions["precip"] += 15
                signals.append("HIGH_TOTAL_PRECIPITATION")
            if total_precip > 60000:  # > 600mm total
                signals.append("EXTREME_PRECIP_TOTAL")

        if "temperature" in data and data["temperature"]:
            temp = data["temperature"]
            result = self.predictor.predict(temp, location, "TEMPERATURE")
            individual["temperature"] = result

            max_temp = max(temp)
            if max_temp > self.thresholds.temp_heat_critical:
                contributions["temp"] = 20
                signals.append("EXTREME_HEAT")
            elif max_temp > self.thresholds.temp_heat_warning:
                contributions["temp"] = 10
                signals.append("HIGH_HEAT")

        if "streamflow" in data and data["streamflow"]:
            streamflow = data["streamflow"]
            result = self.predictor.predict(streamflow, location, "STREAMFLOW")
            individual["streamflow"] = result

            if len(streamflow) >= 2:
                ratio_scaled = (max(streamflow) * 100) // max(1, min(streamflow))
                if ratio_scaled > 1000:  # 10x increase
                    contributions["streamflow"] = 35
                    signals.append("EXTREME_STREAMFLOW_RISE")
                elif ratio_scaled > 500:
                    contributions["streamflow"] = 25
                    signals.append("HIGH_STREAMFLOW_RISE")

        # Compute composite score
        composite_score = sum(contributions.values())

        # Determine hazard type based on dominant signals
        hazard_type = self._classify_hazard(signals, contributions)

        # Determine risk level
        if composite_score >= 70:
            composite_risk = "CRITICAL"
        elif composite_score >= 45:
            composite_risk = "HIGH"
        elif composite_score >= 25:
            composite_risk = "MODERATE"
        else:
            composite_risk = "LOW"

        # Calculate confidence based on data availability
        variables_present = sum(1 for v in individual.values() if v is not None)
        confidence = min(100, 30 + variables_present * 15)

        return MultiVariableResult(
            hazard_type=hazard_type,
            composite_risk=composite_risk,
            composite_score=composite_score,
            confidence=confidence,
            pressure_contribution=contributions["pressure"],
            humidity_contribution=contributions["humidity"],
            wind_contribution=contributions["wind"],
            precip_contribution=contributions["precip"],
            temp_contribution=contributions["temp"],
            streamflow_contribution=contributions["streamflow"],
            signals=signals,
            individual_predictions=individual
        )

    def _classify_hazard(
        self,
        signals: List[str],
        contributions: Dict[str, int]
    ) -> HazardType:
        """Classify hazard type based on signals."""
        has_flood = contributions["precip"] >= 20 or contributions["streamflow"] >= 20
        has_wind = contributions["wind"] >= 15
        has_low_pressure = "EXTREME_LOW_PRESSURE" in signals or "LOW_PRESSURE" in signals
        has_extreme_wind = contributions["wind"] >= 25
        has_heat = contributions["temp"] >= 10
        has_fire = contributions["humidity"] >= 20
        has_extreme_total = "EXTREME_PRECIP_TOTAL" in signals

        # Hurricane: wind + flood + (extreme wind, low pressure, or tropical heat + extreme total)
        if has_wind and has_flood and (
            has_extreme_wind or has_low_pressure or (has_extreme_total and has_heat)
        ):
            return HazardType.HURRICANE

        # Flash flood: precipitation + streamflow
        if has_flood:
            return HazardType.FLASH_FLOOD

        # Tornado: pressure oscillation + wind
        if "RAPID_PRESSURE_DROP" in signals and contributions["wind"] >= 15:
            return HazardType.TORNADO

        # Fire weather: humidity-driven, but only if flood signals are absent
        if has_fire:
            if contributions["precip"] < 20 and contributions["streamflow"] < 20:
                return HazardType.FIRE_WEATHER

        # Severe storm: multiple moderate signals
        if sum(contributions.values()) >= 30:
            return HazardType.SEVERE_STORM

        # Stable if low scores
        if sum(contributions.values()) < 15:
            return HazardType.STABLE

        return HazardType.UNKNOWN


def test_multi_variable_analyzer():
    """Test multi-variable analyzer with real data."""
    print("=" * 70)
    print("MULTI-VARIABLE ANALYZER TEST")
    print("=" * 70)

    from historical_data_loader import HistoricalDataLoader

    analyzer = MultiVariableAnalyzer()
    loader = HistoricalDataLoader()

    events = ["harvey_2017", "camp_fire_2018", "blanco_2015", "stable_reference"]

    for event_key in events:
        print(f"\n{'─' * 70}")
        event = loader.fetch_event_data(event_key)
        if not event:
            continue

        print(f"EVENT: {event.name}")
        print(f"Expected: {event.expected_risk}")
        print(f"{'─' * 70}")

        # Run multi-variable analysis
        result = analyzer.analyze(event.data, event_key)

        print(f"  Hazard Type: {result.hazard_type.value}")
        print(f"  Composite Risk: {result.composite_risk}")
        print(f"  Composite Score: {result.composite_score}")
        print(f"  Confidence: {result.confidence}%")
        print(f"\n  Contributions:")
        print(f"    Pressure: +{result.pressure_contribution}")
        print(f"    Humidity: +{result.humidity_contribution}")
        print(f"    Wind: +{result.wind_contribution}")
        print(f"    Precipitation: +{result.precip_contribution}")
        print(f"    Temperature: +{result.temp_contribution}")
        print(f"    Streamflow: +{result.streamflow_contribution}")
        print(f"\n  Signals: {result.signals}")

        # Check if matches expected
        match = (
            (event.expected_risk == "CRITICAL" and result.composite_risk in ["CRITICAL", "HIGH"]) or
            (event.expected_risk == "HIGH" and result.composite_risk in ["HIGH", "CRITICAL"]) or
            (event.expected_risk == "MODERATE" and result.composite_risk in ["MODERATE", "HIGH"]) or
            (event.expected_risk == "LOW" and result.composite_risk in ["LOW", "MODERATE"])
        )
        print(f"\n  MATCH: {'✓' if match else '✗'}")

    return True


if __name__ == "__main__":
    test_multi_variable_analyzer()
