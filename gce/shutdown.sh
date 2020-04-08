#!/bin/bash

echo "~~~~~~~ Running shutdown script ~~~~~~~"

PID="$(pgrep -o "python")"
# if [[ PID -ne 0 ]]; then
#   echo "Python not running, shutting down immediately."
#   exit 0
# fi

curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"
preempted=$?

echo "Got preempted: $preempted"

if $preempted; then
  echo "Detected pre-emption"

  echo "Send SIGTERM to python PID $PID"
  kill "$PID"
  echo "SIGTERM sent"

  # Give the python script as much time as possible to cleanup
  sleep infinity
fi

echo "~~~~~~~ Completed shutdown script ~~~~~~~"