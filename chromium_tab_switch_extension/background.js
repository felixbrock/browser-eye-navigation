const STATE_URL = "http://127.0.0.1:8766/state";
const POLL_MS = 220;
const MIN_CONFIDENCE = 0.20;
const SWITCH_COOLDOWN_MS = 500;

let switchingEnabled = false;
let lastSwitchAt = 0;
let lastTargetIndex = -1;

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

async function switchToPredictedTab(predictedTab, predictedPosition, tabCountFromState) {
  const tabs = await chrome.tabs.query({ currentWindow: true });
  if (!tabs || tabs.length === 0) {
    return;
  }
  const ordered = [...tabs].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const limit = ordered.length;
  let targetIndex = 0;

  if (Number.isFinite(predictedPosition) && predictedPosition >= 0) {
    targetIndex = Math.floor(predictedPosition * limit);
  } else if (Number.isFinite(predictedTab) && predictedTab >= 1) {
    const modelCount = Number(tabCountFromState) || limit;
    const centerPos = (predictedTab - 0.5) / Math.max(1, modelCount);
    targetIndex = Math.floor(centerPos * limit);
  }
  targetIndex = Math.max(0, Math.min(limit - 1, targetIndex));
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
  const predictedPosition = Number(state.predicted_position);
  const conf = Number(state.confidence || 0);
  if ((predictedTab < 1 && !Number.isFinite(predictedPosition)) || conf < MIN_CONFIDENCE) {
    return;
  }

  const t = nowMs();
  if (t - lastSwitchAt < SWITCH_COOLDOWN_MS) {
    return;
  }

  const tabs = await chrome.tabs.query({ currentWindow: true });
  if (!tabs || tabs.length === 0) {
    return;
  }
  const liveCount = tabs.length;
  let desiredIndex = -1;
  if (Number.isFinite(predictedPosition) && predictedPosition >= 0) {
    desiredIndex = Math.floor(predictedPosition * liveCount);
  } else {
    desiredIndex = predictedTab - 1;
  }
  desiredIndex = Math.max(0, Math.min(liveCount - 1, desiredIndex));
  if (desiredIndex === lastTargetIndex) {
    return;
  }

  try {
    await switchToPredictedTab(predictedTab, predictedPosition, Number(state.tab_count || 0));
    lastSwitchAt = t;
    lastTargetIndex = desiredIndex;
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
    lastTargetIndex = -1;
  }
  await setUi();
});

chrome.action.onClicked.addListener(async () => {
  switchingEnabled = !switchingEnabled;
  if (!switchingEnabled) {
    lastTargetIndex = -1;
  }
  await setUi();
});

setInterval(() => {
  tick();
}, POLL_MS);

setUi();
