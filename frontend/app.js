const timestampEl = document.getElementById("timestamp");
const counterEls = Array.from(document.querySelectorAll("[data-count]"));

function updateTimestamp() {
  if (!timestampEl) return;
  const stamp = new Date().toISOString().replace("T", " ").replace("Z", " UTC");
  timestampEl.textContent = stamp;
}

function animateCounter(element) {
  const target = Number.parseInt(element.dataset.count, 10);
  if (Number.isNaN(target)) return;

  const duration = 1200;
  const start = performance.now();

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const value = Math.floor(progress * target);
    element.textContent = value.toString();
    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  }

  requestAnimationFrame(tick);
}

updateTimestamp();
setInterval(updateTimestamp, 1000);

counterEls.forEach((el) => animateCounter(el));
