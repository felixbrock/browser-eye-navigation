const STATE_URL = "http://127.0.0.1:8766/state";
const POLL_MS = 220;
const MIN_CONFIDENCE = 0.20;
const SWITCH_COOLDOWN_MS = 500;

let switchingEnabled = false;
let lastSwitchAt = 0;
let lastTargetTab = 0;

async function fetchState() {
  const resp = await fetch(STATE_URL, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`state fetch failed: ${resp.status}`);
  }
  return await resp.json();
}

function nowMs() {
  return Date.now();
}

async function setUi() {
  const text = switchingEnabled ? "ON" : "OFF";
  const title = switchingEnabled ? "Tab switch: ON" : "Tab switch: OFF";
  try {
    await chrome.action.setBadgeText({ text });
    await chrome.action.setBadgeBackgroundColor({ color: switchingEnabled ? "#1d9d39" : "#555" });
    await chrome.action.setTitle({ title });
  } catch (_err) {
    // Ignore UI failures.
  }
}

async function switchToPredictedTab(predictedTab, tabCountFromState) {
  const tabs = await chrome.tabs.query({ currentWindow: true });
  if (!tabs || tabs.length === 0) {
    return;
  }
  const ordered = [...tabs].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const limit = Math.min(ordered.length, Number(tabCountFromState) || ordered.length);
  const targetIndex = Math.max(0, Math.min(limit - 1, Number(predictedTab) - 1));
  const target = ordered[targetIndex];
  if (!target || !target.id) {
    return;
  }

  const active = ordered.find((t) => t.active);
  if (active && active.id === target.id) {
    return;
  }

  await chrome.tabs.update(target.id, { active: true });
}

async function tick() {
  if (!switchingEnabled) {
    return;
  }
  let state;
  try {
    state = await fetchState();
  } catch (_err) {
    return;
  }

  if (!state || !state.running) {
    return;
  }
  if (!state.browser_active) {
    return;
  }

  const predictedTab = Number(state.predicted_tab || 0);
  const conf = Number(state.confidence || 0);
  if (predictedTab < 1 || conf < MIN_CONFIDENCE) {
    return;
  }

  const t = nowMs();
  if (t - lastSwitchAt < SWITCH_COOLDOWN_MS) {
    return;
  }

  if (predictedTab === lastTargetTab) {
    return;
  }

  try {
    await switchToPredictedTab(predictedTab, Number(state.tab_count || 0));
    lastSwitchAt = t;
    lastTargetTab = predictedTab;
  } catch (_err) {
    // Ignore switching errors.
  }
}

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "toggle-tab-switch") {
    return;
  }
  switchingEnabled = !switchingEnabled;
  if (!switchingEnabled) {
    lastTargetTab = 0;
  }
  await setUi();
});

chrome.action.onClicked.addListener(async () => {
  switchingEnabled = !switchingEnabled;
  if (!switchingEnabled) {
    lastTargetTab = 0;
  }
  await setUi();
});

setInterval(() => {
  tick();
}, POLL_MS);

setUi();
