const COLLECTOR_URL = "http://127.0.0.1:8767/tab-event";
const TAB_STRIP_LEFT_INSET_PX = 32;
const TAB_STRIP_RIGHT_INSET_PX = 132;
const TAB_STRIP_HEIGHT_PX = 36;
const PINNED_TAB_WIDTH_PX = 44;

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
    const centerPx = leftPx + (width / 2);
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

    const layout = buildTabLayout(win, tabs);
    const payload = {
      event_type: "tab_activated",
      event_ts_ms: Date.now(),
      browser_hint: browserHint(),
      chrome_window_id: activeInfo.windowId,
      tab_id: activeInfo.tabId,
      tab_index: Number(activeTab?.index ?? 0),
      tab_count: tabs.length,
      tab_title: String(activeTab?.title || ""),
      tab_strip: layout.tab_strip,
      tab_candidates: layout.tab_candidates,
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
