const COLLECTOR_URL = "http://127.0.0.1:8767/tab-event";

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

async function postTabEvent(activeInfo) {
  try {
    const [win, tabs, activeTab] = await Promise.all([
      chrome.windows.get(activeInfo.windowId),
      chrome.tabs.query({ windowId: activeInfo.windowId }),
      chrome.tabs.get(activeInfo.tabId),
    ]);

    if (!win || !win.focused) {
      return;
    }

    const payload = {
      event_type: "tab_activated",
      event_ts_ms: Date.now(),
      browser_hint: browserHint(),
      chrome_window_id: activeInfo.windowId,
      tab_id: activeInfo.tabId,
      tab_index: Number(activeTab?.index ?? 0),
      tab_count: tabs.length,
      trigger: "activation_observed",
    };

    await fetch(COLLECTOR_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (_err) {
    // Ignore collector or browser API failures.
  }
}

chrome.tabs.onActivated.addListener((activeInfo) => {
  postTabEvent(activeInfo);
});
