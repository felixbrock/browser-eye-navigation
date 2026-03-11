(function () {
  "use strict";

  document.addEventListener(
    "keydown",
    (event) => {
      if (!event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) {
        return;
      }
      if (event.code !== "KeyB") {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();

      console.log("[Browser Eye Navigation] Ctrl+B pressed");

      chrome.runtime.sendMessage({
        type: "toggle-autoplay",
      });
    },
    true,
  );
})();
