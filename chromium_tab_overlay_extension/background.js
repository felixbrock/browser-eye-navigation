const STATE_URL = "http://127.0.0.1:8765/state";
const POLL_MS = 120;

const MASK_TITLE = "·";
const TARGET_TITLE = "●";
const MASK_COLOR = "#6a6a6a";
const TARGET_COLOR = "#ff2b2b";

let lastEnabled = null;
let lastActiveTabId = null;
let lastTargetTab = 0;

function isScriptableUrl(url) {
  if (!url) return false;
  return (
    url.startsWith("http://") ||
    url.startsWith("https://") ||
    url.startsWith("file://")
  );
}

function dotSvgDataUrl(color) {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'><circle cx='32' cy='32' r='20' fill='${color}'/></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

async function fetchState() {
  const resp = await fetch(STATE_URL, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`state fetch failed: ${resp.status}`);
  }
  return await resp.json();
}

async function tabsInCurrentWindow() {
  return await chrome.tabs.query({ currentWindow: true });
}

async function activeTabInCurrentWindow() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs && tabs.length > 0 ? tabs[0] : null;
}

async function activateTargetTab(targetTab) {
  const tabs = await tabsInCurrentWindow();
  if (!tabs || tabs.length === 0) return;
  const ordered = [...tabs].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const targetIndex = Math.max(0, Math.min(ordered.length - 1, Number(targetTab) - 1));
  const target = ordered[targetIndex];
  if (!target || !target.id) return;
  if (target.active) return;
  try {
    await chrome.tabs.update(target.id, { active: true });
  } catch (_err) {
    // Ignore switching errors.
  }
}

async function injectSetVisual(tabId, titleText, faviconUrl) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      args: [titleText, faviconUrl],
      func: (nextTitle, nextFavicon) => {
        if (window.__tabCalibOrigTitle === undefined) {
          window.__tabCalibOrigTitle = document.title || "";
        }

        const existingIcon = document.querySelector("link[rel*='icon']");
        if (window.__tabCalibOrigFavicon === undefined) {
          window.__tabCalibOrigFavicon = existingIcon ? existingIcon.href : "";
        }

        document.title = nextTitle;

        let iconEl = existingIcon;
        if (!iconEl) {
          iconEl = document.createElement("link");
          iconEl.rel = "icon";
          document.head.appendChild(iconEl);
        }
        iconEl.href = nextFavicon;
      },
    });
  } catch (_err) {
    // Ignore unscriptable tabs.
  }
}

async function injectRestore(tabId) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        if (window.__tabCalibOrigTitle !== undefined) {
          document.title = window.__tabCalibOrigTitle;
          delete window.__tabCalibOrigTitle;
        }

        const iconEl = document.querySelector("link[rel*='icon']");
        if (window.__tabCalibOrigFavicon !== undefined) {
          if (iconEl) {
            if (window.__tabCalibOrigFavicon) {
              iconEl.href = window.__tabCalibOrigFavicon;
            } else {
              iconEl.remove();
            }
          }
          delete window.__tabCalibOrigFavicon;
        }
      },
    });
  } catch (_err) {
    // Ignore unscriptable tabs.
  }
}

async function applyCalibrationVisuals(enabled) {
  const allTabs = await tabsInCurrentWindow();
  const activeTab = await activeTabInCurrentWindow();
  const activeId = activeTab && activeTab.id ? activeTab.id : null;

  if (!enabled) {
    for (const tab of allTabs) {
      if (!tab.id || !isScriptableUrl(tab.url)) continue;
      await injectRestore(tab.id);
    }
    lastActiveTabId = null;
    return;
  }

  const maskIcon = dotSvgDataUrl(MASK_COLOR);
  const targetIcon = dotSvgDataUrl(TARGET_COLOR);

  for (const tab of allTabs) {
    if (!tab.id || !isScriptableUrl(tab.url)) continue;
    const isActive = activeId !== null && tab.id === activeId;
    const title = isActive ? TARGET_TITLE : MASK_TITLE;
    const icon = isActive ? targetIcon : maskIcon;
    await injectSetVisual(tab.id, title, icon);
  }

  lastActiveTabId = activeId;
}

async function tick() {
  let enabled = false;
  let targetTab = 0;
  try {
    const state = await fetchState();
    enabled = Boolean(state && state.enabled);
    targetTab = Number(state?.target_tab || 0);
  } catch (_err) {
    enabled = false;
    targetTab = 0;
  }

  if (enabled && targetTab >= 1) {
    await activateTargetTab(targetTab);
  }

  const activeTab = await activeTabInCurrentWindow();
  const activeId = activeTab && activeTab.id ? activeTab.id : null;
  const shouldRefresh =
    lastEnabled !== enabled ||
    lastTargetTab !== targetTab ||
    (enabled && activeId !== lastActiveTabId);

  if (shouldRefresh) {
    await applyCalibrationVisuals(enabled);
    lastEnabled = enabled;
    lastTargetTab = targetTab;
  }
}

setInterval(() => {
  tick();
}, POLL_MS);

chrome.tabs.onActivated.addListener(() => {
  tick();
});

chrome.tabs.onUpdated.addListener(() => {
  tick();
});

chrome.windows.onFocusChanged.addListener(() => {
  tick();
});

tick();
