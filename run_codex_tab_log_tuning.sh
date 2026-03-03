#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_tab_log_tuning.sh --log-file <path> [--codex-cmd <cmd>] [--dry-run]

Description:
  Starts a Codex tuning pass focused only on tab-classification files using
  a tab calibration log as context.
EOF
}

log_file=""
codex_cmd="codex"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      [[ $# -ge 2 ]] || { echo "ERROR: --log-file requires a value" >&2; exit 2; }
      log_file="$2"
      shift 2
      ;;
    --codex-cmd)
      [[ $# -ge 2 ]] || { echo "ERROR: --codex-cmd requires a value" >&2; exit 2; }
      codex_cmd="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$log_file" ]] || { echo "ERROR: --log-file is required" >&2; usage >&2; exit 2; }
[[ -f "$log_file" ]] || { echo "ERROR: log file not found: $log_file" >&2; exit 1; }
abs_log_file="$(realpath "$log_file")"

if ! command -v "$codex_cmd" >/dev/null 2>&1; then
  echo "ERROR: codex command not found: $codex_cmd" >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: must be run from inside a git repository" >&2
  exit 1
fi

if [[ "$dry_run" -ne 1 && -n "$(git status --porcelain)" ]]; then
  echo "ERROR: git working tree is not clean. Commit or stash existing changes first." >&2
  exit 1
fi

tmp_prompt="$(mktemp)"
cleanup() { rm -f "$tmp_prompt"; }
trap cleanup EXIT

{
  cat <<EOF
You are in /home/felix/repos/eye-tracker.

Task:
1. Analyze the tab-calibration log below.
2. Improve tab classification accuracy and robustness.
3. Modify only tab-focused files:
   - model.py
   - calibration.py
   - tracker.py
   - test.py
   - chromium_tab_overlay_extension/*
   - chromium_tab_switch_extension/*
4. Keep existing eye_tracker.py/calibration.py and train flow untouched.
5. Keep browser integration aligned with current tab workflow:
   - one-time startup focus only (no forced refocus loops)
   - overlay state publisher endpoint semantics used by calibration/test
   - switch-state publisher endpoint semantics used by tracker
6. Run quick checks (at least syntax checks).
7. Summarize what changed and why.

Tab calibration log path: $abs_log_file
Tab calibration log content:
EOF
  echo '```json'
  cat "$abs_log_file"
  echo
  echo '```'
} > "$tmp_prompt"

if [[ "$dry_run" -eq 1 ]]; then
  cat "$tmp_prompt"
  exit 0
fi

"$codex_cmd" exec - < "$tmp_prompt"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No repo changes detected; no commit created."
  exit 0
fi

python3 -m py_compile model.py calibration.py tracker.py test.py
python3 - <<'PY'
import json
for p in (
    "chromium_tab_overlay_extension/manifest.json",
    "chromium_tab_switch_extension/manifest.json",
):
    with open(p, "r", encoding="utf-8") as f:
        json.load(f)
print("extension manifests validated")
PY
git add -A
git commit -m "run tab log tuning (codex context)" \
  -m "Log file: ${abs_log_file}"
git push
echo "Committed and pushed tab-tuning changes."
