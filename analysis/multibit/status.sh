#!/bin/bash
# Status of every accumulation sweep: running processes, live progress, finished CSVs.
cd "$(dirname "$0")/../.." || exit 1

echo "=== running ==="
ps -eo pid,etime,cmd --no-headers \
  | grep -E "accum_sweep|kill -0" | grep -v grep || echo "  none"

echo
echo "=== progress ==="
for f in results/accum_*.log; do
  [ -f "$f" ] || continue
  # tqdm separates updates with \r, so split on it and take the newest non-blank
  # line; grepping for a pattern can miss a partially flushed final write.
  line=$(tr '\r' '\n' < "$f" | grep -v '^[[:space:]]*$' | tail -1)
  printf '  %-26s %s\n' "$(basename "$f" .log)" "${line:-(starting up)}"
done

echo
echo "=== finished CSVs ==="
for f in results/accum_*.csv; do
  [ -f "$f" ] || continue
  printf '  %-54s %5d injections\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
done
