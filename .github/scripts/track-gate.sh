#!/usr/bin/env bash
# Runs one gate under signal tracing and streams diagnostics to the step log.
#
# Usage: track-gate.sh <gate> [cargo ci args...]
#
# Everything is written to stdout/stderr as it happens, so the lines before a
# runner shutdown survive in the step log even when no artifact is uploaded:
# - strace: every kill/tkill/tgkill/pidfd_send_signal call in the gate's
#   process tree, and every SIGTERM/SIGINT/SIGHUP delivery with its sender
#   (si_pid), timestamped.
# - watchdog: every 10 s, memory, disk, the busiest processes and the newest
#   mutant log, which names the mutant in flight.
# - on SIGTERM to this wrapper or on exit: the process tree, kernel OOM lines
#   and the decoded exit status.
set -u

gate="$1"
shift
out_dir="target/ci-artifacts/${gate}.out"

stamp() { date -u +%H:%M:%S.%N | cut -c1-12; }

snapshot() {
  echo "--- watchdog $(stamp) [$1]"
  free -m | sed -n 2p
  df -h / | tail -1
  ps -eo pid,ppid,pgid,pcpu,pmem,etime,args --sort=-pcpu | cut -c1-160 | head -8
  find "$out_dir" -name '*.log' -printf '%T@ %p\n' 2>/dev/null |
    sort -n | tail -1 | cut -d' ' -f2
}

post_mortem() {
  echo "=== post-mortem $(stamp) [$1]"
  ps -eo pid,ppid,pgid,user,etime,args --forest | cut -c1-200
  sudo dmesg 2>/dev/null | grep -iE 'oom|killed process' | tail -20
}

(
  while true; do
    snapshot periodic
    sleep 10
  done
) &
watchdog=$!

strace_opts=(-f -qq -tt --seccomp-bpf
  -e trace=kill,tkill,tgkill,pidfd_send_signal
  -e signal=SIGTERM,SIGINT,SIGHUP)
# Name each pid by its command where this strace supports it.
if strace -Y -e trace=none true 2>/dev/null; then
  strace_opts+=(-Y)
fi

strace "${strace_opts[@]}" cargo ci "$@" --only "$gate" &
gate_pid=$!

on_term() {
  echo "=== wrapper received SIGTERM at $(stamp)"
  post_mortem sigterm
  kill "$watchdog" 2>/dev/null
  exit 143
}
trap on_term TERM

rc=0
wait "$gate_pid" || rc=$?
kill "$watchdog" 2>/dev/null

if [ "$rc" -gt 128 ]; then
  echo "=== gate exited with $rc (signal $((rc - 128)))"
  post_mortem "exit $rc"
else
  echo "=== gate exited with $rc"
fi
exit "$rc"
