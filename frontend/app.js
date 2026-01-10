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
const gateStatusEl = document.getElementById("gateStatus");
const gateEventsEl = document.getElementById("gateEvents");
const gateRiskEl = document.getElementById("gateRisk");
const gateHazardEl = document.getElementById("gateHazard");
const gateLeadEl = document.getElementById("gateLead");
const gateGridEl = document.getElementById("gateGrid");
const gateTableEl = document.getElementById("gateTable");
const gateNoteEl = document.getElementById("gateNote");

const sparklineBars = Array.from(document.querySelectorAll("#sparkline span"));
const hazardSelect = document.getElementById("hazardSelect");

const gauntletSources = [
  {
    label: "predictive_gauntlet_live_report.json",
    path: "../predictive_gauntlet_live_report.json"
  },
  {
    label: "predictive_gauntlet_report.json",
    path: "../predictive_gauntlet_report.json"
  }
];

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

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(1)}%`;
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

function updateGateStatus(passed) {
  if (!gateStatusEl) return;
  if (passed === null) {
    gateStatusEl.textContent = "NO DATA";
    gateStatusEl.className = "tag neutral";
    return;
  }
  gateStatusEl.textContent = passed ? "PASS" : "FAIL";
  gateStatusEl.className = `tag ${passed ? "gate-pass" : "gate-fail"}`;
}

function buildGateState(ok, label) {
  const pill = document.createElement("span");
  pill.className = "state";
  if (ok === null) {
    pill.classList.add("warn");
    pill.textContent = label || "N/A";
    return pill;
  }
  pill.classList.add(ok ? "ok" : "hot");
  pill.textContent = label || (ok ? "PASS" : "FAIL");
  return pill;
}

function renderGateGrid(gates) {
  if (!gateGridEl) return;
  gateGridEl.innerHTML = "";
  Object.entries(gates).forEach(([metric, gate]) => {
    const item = document.createElement("div");
    item.className = "gate-item";

    const label = document.createElement("span");
    label.textContent = metric.replace(/_/g, " ");

    const value = buildGateState(
      gate.pass,
      `${gate.value.toFixed(1)}% / ${gate.threshold}%`
    );

    item.append(label, value);
    gateGridEl.appendChild(item);
  });
}

function renderGateTable(results) {
  if (!gateTableEl) return;
  gateTableEl.innerHTML = "";
  results.forEach((result) => {
    const row = document.createElement("div");
    row.className = "gate-row";

    const name = document.createElement("span");
    name.className = "gate-name";
    name.textContent = result.name || result.id || "UNKNOWN";

    const riskLabel = result.risk_ok ? result.predicted_risk : "RISK MISS";
    const hazardLabel = result.hazard_expected
      ? result.hazard_predicted || "NO HAZARD"
      : "HAZARD N/A";
    const leadLabel = result.lead_score !== null
      ? `LEAD ${result.lead_score}`
      : "LEAD N/A";

    row.append(
      name,
      buildGateState(result.risk_ok, riskLabel),
      buildGateState(result.hazard_expected ? result.hazard_ok : null, hazardLabel),
      buildGateState(result.lead_score !== null ? result.lead_ok : null, leadLabel)
    );

    gateTableEl.appendChild(row);
  });
}

async function loadGauntletReport() {
  if (gateNoteEl) {
    gateNoteEl.textContent = "Loading gauntlet report...";
  }
  let report = null;
  let sourceLabel = "";

  for (const source of gauntletSources) {
    try {
      const response = await fetch(source.path, { cache: "no-store" });
      if (!response.ok) {
        continue;
      }
      report = await response.json();
      sourceLabel = source.label;
      break;
    } catch (err) {
      continue;
    }
  }

  if (!report) {
    updateGateStatus(null);
    if (gateNoteEl) {
      gateNoteEl.textContent = "Gauntlet report not available.";
    }
    return;
  }

  if (gateNoteEl) {
    gateNoteEl.textContent = `Source: ${sourceLabel}`;
  }

  gateEventsEl.textContent = report.events_tested ?? "--";
  gateRiskEl.textContent = formatPercent(report.risk_accuracy);
  gateHazardEl.textContent = formatPercent(report.hazard_accuracy);
  gateLeadEl.textContent = formatPercent(report.lead_success);

  const gates = report.quality_gates?.gates || {};
  updateGateStatus(report.quality_gates?.passed ?? null);
  renderGateGrid(gates);
  renderGateTable(report.results || []);
}

updateTimestamp();
updateSparkline();
setInterval(updateTimestamp, 1000);
loadGauntletReport();

document.getElementById("runAnalysis")?.addEventListener("click", runRandomAnalysis);
document.getElementById("loadSample")?.addEventListener("click", loadSample);
document.getElementById("loadGauntlet")?.addEventListener("click", loadGauntletReport);
