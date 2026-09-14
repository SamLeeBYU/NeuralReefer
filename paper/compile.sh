#!/usr/bin/env bash
# compile.sh -- Build main.pdf from main.tex
#
# Pipeline: pdflatex -> bibtex -> pdflatex (re-run as needed to settle
# citations and cross-references).
#
# Usage: ./compile.sh
# Run from anywhere -- the script cd's to its own directory first.

set -euo pipefail

BASENAME="main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEX_FILE="${BASENAME}.tex"
LOG_DIR="$(mktemp -d)"
MAX_RERUNS=4

# -- locate the toolchain -------------------------------------------------
# Prefer PATH (native Linux/macOS, or Windows with these on PATH); fall back
# to the known Windows install locations for a WSL session where they are
# not.
find_tool() {
  local name="$1"
  shift
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "ERROR: could not find '$name' on PATH or in known install locations." >&2
  return 1
}

PDFLATEX="$(find_tool pdflatex \
  "/mnt/c/Users/samle/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe" \
  "/mnt/c/Users/samle/AppData/Roaming/TinyTeX/bin/windows/pdflatex.exe")"
BIBTEX="$(find_tool bibtex \
  "/mnt/c/Users/samle/AppData/Local/Programs/MiKTeX/miktex/bin/x64/bibtex.exe" \
  "/mnt/c/Users/samle/AppData/Roaming/TinyTeX/bin/windows/bibtex.exe")"

echo "==> Using pdflatex: $PDFLATEX"
echo "==> Using bibtex:   $BIBTEX"

# -- step 1: LaTeX + BibTeX, re-running until stable -----------------------
run_pdflatex() {
  local pass_log="$1"
  local status=0
  "$PDFLATEX" -interaction=nonstopmode -halt-on-error "$TEX_FILE" \
    > "$pass_log" 2>&1 || status=$?
  if [[ $status -ne 0 ]]; then
    echo "ERROR: pdflatex failed. Last 60 lines of $pass_log:" >&2
    tail -n 60 "$pass_log" >&2
    exit 1
  fi
}

echo "==> [1/3] pdflatex (pass 1) ..."
run_pdflatex "$LOG_DIR/pdflatex_1.log"

echo "==> [2/3] bibtex ..."
"$BIBTEX" "$BASENAME" > "$LOG_DIR/bibtex.log" 2>&1 || {
  echo "ERROR: bibtex failed. See $LOG_DIR/bibtex.log" >&2
  tail -n 40 "$LOG_DIR/bibtex.log" >&2
  exit 1
}

echo "==> [3/3] pdflatex (re-running until citations/cross-references settle) ..."
pass=2
while (( pass <= MAX_RERUNS + 1 )); do
  log="$LOG_DIR/pdflatex_${pass}.log"
  run_pdflatex "$log"
  if grep -qi "Rerun to get \|there were undefined references\|there were undefined citations" "$log"; then
    echo "    pass $pass: cross-references still settling, rerunning ..."
    pass=$(( pass + 1 ))
    continue
  fi
  break
done

if (( pass > MAX_RERUNS + 1 )); then
  echo "WARNING: cross-references may not be fully settled after $MAX_RERUNS reruns." >&2
  echo "         Check $LOG_DIR/pdflatex_${pass}.log" >&2
fi

# -- report -----------------------------------------------------------------
echo ""
if [[ -f "${BASENAME}.pdf" ]]; then
  echo "==> Success: ${BASENAME}.pdf ($(stat -c %s "${BASENAME}.pdf" 2>/dev/null || stat -f %z "${BASENAME}.pdf") bytes)"
else
  echo "ERROR: ${BASENAME}.pdf was not produced." >&2
  exit 1
fi

echo ""
echo "==> Remaining warnings from the final pass:"
grep -i "warning" "$LOG_DIR/pdflatex_${pass}.log" || echo "    (none)"
echo ""
echo "Full logs kept in: $LOG_DIR"
