#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_tab_log_tuning.sh --log-file <path> [--codex-cmd <cmd>] [--dry-run]

Description:
  Starts a Codex tuning pass focused only on tab-classification files using
  a tab calibration log as context.
  Every run writes a reasoning log and replays prior run logs in order.
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
repo_root="$(pwd)"

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

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required" >&2
  exit 1
fi

runs_dir="calibration_logs/train_runs"
history_file="calibration_logs/train_history.md"
run_id="$(date -u +%Y%m%d_%H%M%S)"
run_log_file="$runs_dir/train_run_${run_id}.md"
codex_output_file="$runs_dir/train_run_${run_id}_codex_output.txt"
tmp_prompt="$(mktemp)"
tmp_history_list="$(mktemp)"

cleanup() { rm -f "$tmp_prompt" "$tmp_history_list"; }
trap cleanup EXIT

mkdir -p "$runs_dir"

# Collect previous run logs in chronological order.
find "$runs_dir" -maxdepth 1 -type f -name 'train_run_*.md' | sort > "$tmp_history_list"
history_count="$(wc -l < "$tmp_history_list" | tr -d ' ')"

calibration_summary="$(
python3 - "$abs_log_file" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
with p.open("r", encoding="utf-8") as f:
    payload = json.load(f)
model = payload.get("model", {}) or {}
print(json.dumps({
    "created_at": payload.get("created_at"),
    "tab_count": payload.get("tab_count"),
    "rounds": payload.get("rounds"),
    "trial_count": len(payload.get("trials", []) or []),
    "train_accuracy_in_sample": payload.get("train_accuracy_in_sample"),
    "model_total_samples": model.get("total_samples"),
    "model_rmse": ((model.get("position_model") or {}).get("rmse")),
}, indent=2))
PY
)"

{
  cat <<EOF
# Train Run $run_id

## Metadata
- Run ID: $run_id
- Started UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- Calibration log: $abs_log_file

## Required Reasoning (fill this in during this run)
<!-- TODO_Codex_FILL_REASONING -->
1. Wrong-turn assessment against prior run history and latest calibration:
2. Root-cause hypotheses:
3. Changes made and why:
4. Risks introduced / tradeoffs:
5. Validation performed:
6. Next-step recommendation:
EOF
} > "$run_log_file"

{
  cat <<EOF
You are in $repo_root.

Task:
1. Read historical train reasoning logs in chronological order (oldest to newest).
2. Before coding, perform a "Wrong-Turn Check" comparing latest calibration metrics against the history.
3. Analyze the latest calibration log below.
4. Improve tab classification accuracy and robustness.
5. Modify only tab-focused files plus train logs:
   - model.py
   - calibration.py
   - tracker.py
   - test.py
   - chromium_tab_overlay_extension/*
   - chromium_tab_switch_extension/*
   - $run_log_file
   - $history_file
6. Keep non-tab legacy flows untouched.
7. Keep browser integration aligned with current tab workflow:
   - one-time startup focus only (no forced refocus loops)
   - overlay state publisher endpoint semantics used by calibration/test
   - switch-state publisher endpoint semantics used by tracker
8. Run quick checks (at least syntax checks).
9. Update $run_log_file with concrete reasoning:
   - Must replace the TODO marker.
   - Must include explicit wrong-turn assessment at the beginning.
   - Must connect the latest calibration metrics to specific code changes.
10. If prior history indicates a likely wrong turn, explicitly say it in the log before any solution details.

Latest calibration summary:
$calibration_summary

Historical train logs (ordered oldest -> newest):
EOF
  if [[ "$history_count" -eq 0 ]]; then
    echo "(none yet)"
  else
    while IFS= read -r hist_log; do
      [[ -n "$hist_log" ]] || continue
      echo
      echo "--- BEGIN HISTORY LOG: $hist_log ---"
      cat "$hist_log"
      echo
      echo "--- END HISTORY LOG: $hist_log ---"
    done < "$tmp_history_list"
  fi
  cat <<EOF

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

"$codex_cmd" exec - < "$tmp_prompt" | tee "$codex_output_file"

if grep -q 'TODO_Codex_FILL_REASONING' "$run_log_file"; then
  echo "ERROR: reasoning log was not completed: $run_log_file" >&2
  exit 1
fi

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

{
  echo
  echo "## Machine Summary"
  echo "- Completed UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "- Latest calibration log: $abs_log_file"
  echo "- Codex output log: $codex_output_file"
  echo "- Changed files:"
  git diff --name-status
} >> "$run_log_file"

{
  echo "## $run_id"
  echo "- calibration_log: $abs_log_file"
  echo "- run_log: $run_log_file"
  echo "- codex_output: $codex_output_file"
  echo "- changed_files:"
  git diff --name-status | sed 's/^/  - /'
  echo
} >> "$history_file"

git add -A
git commit -m "run tab log tuning (codex context)" \
  -m "Log file: ${abs_log_file}" \
  -m "Train reasoning log: ${run_log_file}"
git push
echo "Committed and pushed tab-tuning changes."
echo "Run reasoning log: $run_log_file"
echo "Train history index: $history_file"
