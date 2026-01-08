const riskDial = document.getElementById("riskDial");
const riskScoreEl = document.getElementById("riskScore");
const riskLevelEl = document.getElementById("riskLevel");
const hazardTypeEl = document.getElementById("hazardType");
const confidenceEl = document.getElementById("confidenceValue");
const primarySignalEl = document.getElementById("primarySignal");
const attractorEl = document.getElementById("attractorValue");
const lyapunovEl = document.getElementById("lyapunovValue");
const trendEl = document.getElementById("trendValue");
const signalTagsEl = document.getElementById("signalTags");
const streamMetaEl = document.getElementById("streamMeta");
const timestampEl = document.getElementById("timestamp");

const sparklineBars = Array.from(document.querySelectorAll("#sparkline span"));
const hazardSelect = document.getElementById("hazardSelect");

const hazardPool = [
  "FLASH_FLOOD",
  "FIRE_WEATHER",
  "HURRICANE",
  "TORNADO",
  "SEVERE_STORM",
  "STABLE"
];

const attractorPool = [
  "CLEAR",
  "WATCH",
  "STEADY_RAIN",
  "FLASH_FLOOD",
  "TORNADO"
];

const trendPool = ["RISING", "FALLING", "OSCILLATING", "STABLE"];
const lyapunovPool = ["CHAOTIC", "HIGHLY_CHAOTIC", "MARGINALLY_STABLE"];
const primaryPool = ["PRESSURE", "STREAMFLOW", "PRECIPITATION", "HUMIDITY", "WIND"];

const signalSets = [
  ["HIGH_FIRE_DANGER_HUMIDITY", "HIGH_WIND", "EXTREME_STREAMFLOW_RISE"],
  ["RAPID_PRESSURE_DROP", "HIGH_WIND", "LOW_PRESSURE"],
  ["EXTREME_PRECIPITATION_RATE", "HIGH_TOTAL_PRECIPITATION", "HIGH_STREAMFLOW_RISE"],
  ["OSCILLATION_SPIKE", "GUST_FRONT", "CHAOS_THRESHOLD"],
  ["THERMAL_SURGE", "HIGH_HEAT", "LOW_HUMIDITY"]
];

function updateTimestamp() {
  const stamp = new Date().toISOString().replace("T", " ").replace("Z", " UTC");
  timestampEl.textContent = stamp;
}

function riskLevelFromScore(score) {
  if (score >= 70) return "CRITICAL";
  if (score >= 50) return "HIGH";
  if (score >= 25) return "MODERATE";
  return "LOW";
}

function setRisk(score) {
  const clamped = Math.max(0, Math.min(score, 100));
  riskDial.style.setProperty("--risk", clamped);
  riskScoreEl.textContent = clamped;
  const level = riskLevelFromScore(clamped);
  riskLevelEl.textContent = level;
  riskLevelEl.className = `tag risk-${level.toLowerCase()}`;
}

function randomChoice(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function updateSparkline() {
  const values = sparklineBars.map(() => Math.floor(Math.random() * 80) + 10);
  sparklineBars.forEach((bar, idx) => {
    bar.style.setProperty("--bar", `${values[idx]}%`);
  });
  const min = Math.min(...values);
  const max = Math.max(...values);
  streamMetaEl.textContent = `range ${10000 + min * 5} - ${10100 + max * 5}`;
}

function updateSignals(tags) {
  signalTagsEl.innerHTML = "";
  tags.forEach((tag) => {
    const span = document.createElement("span");
    span.textContent = tag;
    signalTagsEl.appendChild(span);
  });
}

function runRandomAnalysis() {
  const base = hazardSelect?.value || randomChoice(hazardPool);
  const score = Math.floor(Math.random() * 50) + 40;
  const confidence = Math.floor(Math.random() * 20) + 80;

  hazardTypeEl.textContent = base;
  confidenceEl.textContent = `${confidence}%`;
  primarySignalEl.textContent = randomChoice(primaryPool);
  attractorEl.textContent = randomChoice(attractorPool);
  lyapunovEl.textContent = randomChoice(lyapunovPool);
  trendEl.textContent = randomChoice(trendPool);

  setRisk(score);
  updateSignals(randomChoice(signalSets));
  updateSparkline();
}

function loadSample() {
  hazardTypeEl.textContent = "FLASH_FLOOD";
  confidenceEl.textContent = "91%";
  primarySignalEl.textContent = "STREAMFLOW";
  attractorEl.textContent = "FLASH_FLOOD";
  lyapunovEl.textContent = "HIGHLY_CHAOTIC";
  trendEl.textContent = "RISING";
  setRisk(82);
  updateSignals(signalSets[2]);
  updateSparkline();
}

updateTimestamp();
updateSparkline();
setInterval(updateTimestamp, 1000);

document.getElementById("runAnalysis")?.addEventListener("click", runRandomAnalysis);
document.getElementById("loadSample")?.addEventListener("click", loadSample);
