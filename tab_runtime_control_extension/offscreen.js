const POLL_INTERVAL_MS = 250;
const DEBUG = true;

function debugLog(event, details = {}) {
  if (!DEBUG) {
    return;
  }
  console.log("[Browser Eye Navigation Offscreen]", event, details);
}

setInterval(() => {
  chrome.runtime
    .sendMessage({ type: "runtime-poll" })
    .then((response) => debugLog("runtime_poll_ack", response || {}))
    .catch((error) => debugLog("runtime_poll_send_failed", { error: String(error) }));
}, POLL_INTERVAL_MS);

debugLog("offscreen_started", { pollIntervalMs: POLL_INTERVAL_MS });
chrome.runtime
  .sendMessage({ type: "runtime-poll" })
  .then((response) => debugLog("runtime_poll_initial_ack", response || {}))
  .catch((error) => debugLog("runtime_poll_initial_send_failed", { error: String(error) }));
