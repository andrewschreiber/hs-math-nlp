#!/bin/bash

echo "~~~~~~~ Running shutdown script ~~~~~~~"

preempted = curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"

echo "Got preempted: $preempted"

if $preempted; then
  echo "Detected pre-emption"
  PID="$(pgrep -o "python")"

  echo "Send SIGTERM to python PID $PID"
  kill "$PID"
  echo "SIGTERM sent"

  # Give the python script as much time as possible to cleanup
  sleep infinity
fi

echo "~~~~~~~ Completed shutdown script ~~~~~~~"