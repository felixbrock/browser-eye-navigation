const RUNTIME_URL = "http://127.0.0.1:8768/runtime-window-state";
const RUNTIME_OFFSCREEN_URL = "offscreen.html";
const TAB_STRIP_LEFT_INSET_PX = 32;
const TAB_STRIP_RIGHT_INSET_PX = 132;
const TAB_STRIP_HEIGHT_PX = 36;
const PINNED_TAB_WIDTH_PX = 44;
const SWITCH_THRESHOLD = 0.25;
const DWELL_MS = 300;
const COOLDOWN_MS = 500;
const STORAGE_KEY_AUTOPLAY = "autoplayEnabled";
const DEBUG = true;
let pollCount = 0;
let autoplayEnabled = false;
let autoplayLoaded = false;
let lastSwitchTsMs = 0;
let lastFocusKey = null;
let lastCandidateKey = null;
let lastCandidateSinceMs = 0;

function debugLog(event, details = {}) {
  if (!DEBUG) {
    return;
  }
  console.log("[Browser Eye Navigation Runtime]", event, details);
}

function browserHint() {
  const ua = String(self.navigator?.userAgent || "");
  if (/\bBrave\//i.test(ua)) {
    return "brave";
  }
  if (/\bChrome\//i.test(ua)) {
    return "google-chrome";
  }
  return "chromium";
}

function clamp(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function buildTabLayout(win, tabs) {
  const windowWidth = Math.max(1, Number(win?.width || 0));
  const stripLeft = TAB_STRIP_LEFT_INSET_PX;
  const stripRightInset = clamp(
    Math.round(windowWidth * 0.08),
    TAB_STRIP_RIGHT_INSET_PX,
    Math.max(TAB_STRIP_RIGHT_INSET_PX, Math.round(windowWidth * 0.18)),
  );
  const stripWidth = Math.max(1, windowWidth - stripLeft - stripRightInset);
  const orderedTabs = [...tabs].sort((a, b) => Number(a.index) - Number(b.index));
  const pinnedCount = orderedTabs.filter((tab) => tab.pinned).length;
  const normalCount = Math.max(0, orderedTabs.length - pinnedCount);
  const pinnedWidth = pinnedCount > 0 ? PINNED_TAB_WIDTH_PX : 0;
  const normalWidth =
    normalCount > 0
      ? Math.max(1, (stripWidth - (pinnedCount * pinnedWidth)) / normalCount)
      : 0;

  let cursor = stripLeft;
  const tabCandidates = orderedTabs.map((tab) => {
    const width = Math.max(1, tab.pinned ? pinnedWidth : normalWidth);
    const leftPx = cursor;
    const rightPx = leftPx + width;
    const centerPx = leftPx + width / 2;
    cursor = rightPx;
    return {
      tab_id: tab.id,
      index: Number(tab.index ?? 0),
      title: String(tab.title || ""),
      is_active: Boolean(tab.active),
      is_pinned: Boolean(tab.pinned),
      bounds_px: {
        left: leftPx,
        right: rightPx,
        center: centerPx,
        width,
        top: 0,
        height: TAB_STRIP_HEIGHT_PX,
      },
      bounds_norm: {
        left: clamp((leftPx - stripLeft) / stripWidth, 0, 1),
        right: clamp((rightPx - stripLeft) / stripWidth, 0, 1),
        center: clamp((centerPx - stripLeft) / stripWidth, 0, 1),
        width: clamp(width / stripWidth, 0, 1),
      },
    };
  });

  return {
    tab_strip: {
      left_px: stripLeft,
      right_px: stripLeft + stripWidth,
      width_px: stripWidth,
      height_px: TAB_STRIP_HEIGHT_PX,
      right_inset_px: stripRightInset,
    },
    tab_candidates: tabCandidates,
  };
}

async function ensureOffscreenDocument() {
  debugLog("ensure_offscreen_document_start");
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(RUNTIME_OFFSCREEN_URL)],
  });
  if (contexts.length > 0) {
    debugLog("ensure_offscreen_document_exists", { count: contexts.length });
    return;
  }
  await chrome.offscreen.createDocument({
    url: RUNTIME_OFFSCREEN_URL,
    reasons: ["DOM_SCRAPING"],
    justification: "Keep a lightweight runtime heartbeat active for live local model polling.",
  });
  debugLog("ensure_offscreen_document_created");
}

async function loadAutoplayEnabled() {
  if (autoplayLoaded) {
    return autoplayEnabled;
  }
  const stored = await chrome.storage.local.get(STORAGE_KEY_AUTOPLAY);
  autoplayEnabled = Boolean(stored?.[STORAGE_KEY_AUTOPLAY]);
  autoplayLoaded = true;
  debugLog("autoplay_loaded", { autoplayEnabled });
  return autoplayEnabled;
}

async function setAutoplayEnabled(nextValue) {
  autoplayEnabled = Boolean(nextValue);
  autoplayLoaded = true;
  await chrome.storage.local.set({ [STORAGE_KEY_AUTOPLAY]: autoplayEnabled });
  if (!autoplayEnabled) {
    lastCandidateKey = null;
    lastCandidateSinceMs = 0;
  }
  debugLog("autoplay_updated", { autoplayEnabled });
  return autoplayEnabled;
}

async function fetchFocusedWindowState() {
  const win = await chrome.windows.getLastFocused({ populate: true });
  if (!win?.focused) {
    debugLog("fetch_focused_window_none");
    return null;
  }
  const tabs = Array.isArray(win.tabs) ? win.tabs : [];
  const activeTab = tabs.find((tab) => tab.active) || null;
  if (!activeTab) {
    debugLog("fetch_active_tab_none", { windowId: win.id });
    return null;
  }
  const layout = buildTabLayout(win, tabs);
  return {
    event_type: "runtime_window_state",
    event_ts_ms: Date.now(),
    browser_hint: browserHint(),
    chrome_window_id: win.id,
    window: {
      left: Number(win.left ?? 0),
      top: Number(win.top ?? 0),
      width: Number(win.width ?? 0),
      height: Number(win.height ?? 0),
      focused: Boolean(win.focused),
      state: String(win.state || "normal"),
    },
    active_tab_id: Number(activeTab.id),
    active_tab_index: Number(activeTab.index ?? 0),
    active_tab_title: String(activeTab.title || ""),
    tab_count: tabs.length,
    tab_strip: layout.tab_strip,
    tab_candidates: layout.tab_candidates,
  };
}

async function maybeLogToggleState(activeTabId, result) {
  await loadAutoplayEnabled();
  debugLog("toggle_state_changed", {
    activeTabId,
    autoplayEnabled,
  });
  if (!activeTabId) {
    return;
  }

  const message = autoplayEnabled
    ? "[Browser Eye Navigation] Runtime autoplay is ON."
    : "[Browser Eye Navigation] Runtime autoplay is OFF.";

  try {
    await chrome.scripting.executeScript({
      target: { tabId: Number(activeTabId) },
      world: "MAIN",
      func: (text) => {
        console.log(text);
      },
      args: [message],
    });
    debugLog("toggle_console_log_injected", { activeTabId, autoplayEnabled });
  } catch (_err) {
    debugLog("toggle_console_log_failed", {
      activeTabId,
      autoplayEnabled,
      error: String(_err),
    });
    // Protected pages such as chrome:// cannot be scripted.
  }
}

async function emitConsoleMessage(activeTabId, message) {
  if (!activeTabId) {
    return;
  }
  await chrome.scripting.executeScript({
    target: { tabId: Number(activeTabId) },
    world: "MAIN",
    func: (text) => {
      console.log(text);
    },
    args: [message],
  });
}

function resetFocusTracking(nextFocusKey) {
  if (lastFocusKey === nextFocusKey) {
    return;
  }
  lastFocusKey = nextFocusKey;
  lastCandidateKey = null;
  lastCandidateSinceMs = 0;
}

function computeActivationDecision(payload, result) {
  const currentTsMs = Date.now();
  const focusKey = `${payload.browser_hint}:${payload.chrome_window_id}`;
  const predictedTabId = Number(result?.predicted_tab_id ?? 0);
  const predictedScore = Number(result?.predicted_score ?? 0);
  const activeTabId = Number(payload.active_tab_id ?? 0);

  resetFocusTracking(focusKey);

  if (!autoplayEnabled) {
    return { activateTabId: null, reason: "autoplay_disabled", predictedScore, threshold: SWITCH_THRESHOLD };
  }
  if (!predictedTabId) {
    lastCandidateKey = null;
    lastCandidateSinceMs = 0;
    return { activateTabId: null, reason: "no_prediction", predictedScore, threshold: SWITCH_THRESHOLD };
  }
  if (predictedScore < SWITCH_THRESHOLD) {
    lastCandidateKey = null;
    lastCandidateSinceMs = 0;
    return { activateTabId: null, reason: "below_threshold", predictedScore, threshold: SWITCH_THRESHOLD };
  }
  if (predictedTabId === activeTabId) {
    lastCandidateKey = `${focusKey}:${predictedTabId}`;
    lastCandidateSinceMs = currentTsMs;
    return { activateTabId: null, reason: "already_active", predictedScore, threshold: SWITCH_THRESHOLD };
  }

  const candidateKey = `${focusKey}:${predictedTabId}`;
  if (lastCandidateKey !== candidateKey) {
    lastCandidateKey = candidateKey;
    lastCandidateSinceMs = currentTsMs;
  }

  const dwellMet = currentTsMs - lastCandidateSinceMs >= DWELL_MS;
  const cooldownMet = currentTsMs - lastSwitchTsMs >= COOLDOWN_MS;
  if (!dwellMet) {
    return {
      activateTabId: null,
      reason: "dwell_pending",
      predictedScore,
      threshold: SWITCH_THRESHOLD,
      dwellMsElapsed: currentTsMs - lastCandidateSinceMs,
    };
  }
  if (!cooldownMet) {
    return {
      activateTabId: null,
      reason: "cooldown_pending",
      predictedScore,
      threshold: SWITCH_THRESHOLD,
      cooldownMsElapsed: currentTsMs - lastSwitchTsMs,
    };
  }

  lastSwitchTsMs = currentTsMs;
  return { activateTabId: predictedTabId, reason: "switch", predictedScore, threshold: SWITCH_THRESHOLD };
}

async function pollRuntime() {
  pollCount += 1;
  await loadAutoplayEnabled();
  const payload = await fetchFocusedWindowState();
  if (!payload) {
    debugLog("poll_skipped_no_payload", { pollCount });
    return;
  }

  try {
    const response = await fetch(RUNTIME_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      debugLog("runtime_http_not_ok", {
        pollCount,
        status: response.status,
        statusText: response.statusText,
      });
      return;
    }
    const result = await response.json();
    debugLog("runtime_response_received", {
      pollCount,
      reason: result?.reason,
      predicted_tab_id: result?.predicted_tab_id,
      predicted_index: result?.predicted_index,
      predicted_score: result?.predicted_score,
      autoplay_enabled: autoplayEnabled,
      debug: result?.debug,
    });
    const decision = computeActivationDecision(payload, result);

    const activateTabId = Number(decision.activateTabId ?? 0);
    if (!activateTabId || activateTabId === payload.active_tab_id) {
      debugLog("no_tab_activation", {
        pollCount,
        activeTabId: payload.active_tab_id,
        activateTabId,
        reason: decision.reason,
        predictedScore: decision.predictedScore,
        threshold: decision.threshold,
        dwellMsElapsed: decision.dwellMsElapsed,
        cooldownMsElapsed: decision.cooldownMsElapsed,
        runtimeReason: result?.reason,
      });
      return;
    }
    debugLog("tab_activation_attempt", {
      pollCount,
      chrome_window_id: payload.chrome_window_id,
      fromTabId: payload.active_tab_id,
      toTabId: activateTabId,
      reason: decision.reason,
      predictedScore: decision.predictedScore,
      threshold: decision.threshold,
    });
    await chrome.tabs.update(activateTabId, { active: true });
    debugLog("tab_activation_success", {
      pollCount,
      toTabId: activateTabId,
    });
  } catch (_err) {
    debugLog("runtime_poll_failed", {
      pollCount,
      error: String(_err),
    });
    // Ignore runtime connectivity failures.
  }
}

async function handleToggleAutoplay(tabId) {
  try {
    await loadAutoplayEnabled();
    await setAutoplayEnabled(!autoplayEnabled);
    await maybeLogToggleState(tabId);
    debugLog("shortcut_toggled_autoplay", { tabId, autoplayEnabled });
  } catch (_err) {
    debugLog("shortcut_toggle_failed", {
      tabId,
      error: String(_err),
    });
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "toggle-autoplay") {
    const tabId = Number(_sender?.tab?.id ?? 0);
    debugLog("toggle_autoplay_message_received", { tabId });
    handleToggleAutoplay(tabId)
      .then(() => sendResponse({ ok: true }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  if (message?.type !== "runtime-poll") {
    return false;
  }
  debugLog("runtime_poll_message_received", { pollCount: pollCount + 1 });
  pollRuntime()
    .then(() => sendResponse({ ok: true }))
    .catch(() => sendResponse({ ok: false }));
  return true;
});

chrome.runtime.onInstalled.addListener(() => {
  ensureOffscreenDocument().catch((err) => {
    debugLog("ensure_offscreen_document_install_failed", { error: String(err) });
  });
});

chrome.runtime.onStartup.addListener(() => {
  ensureOffscreenDocument().catch((err) => {
    debugLog("ensure_offscreen_document_startup_failed", { error: String(err) });
  });
});

ensureOffscreenDocument().catch((err) => {
  debugLog("ensure_offscreen_document_boot_failed", { error: String(err) });
});
