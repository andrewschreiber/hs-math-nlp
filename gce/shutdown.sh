#!/bin/bash
# If this doesn't work, make sure your image isn't editing your metadata

echo "~~~~~~~ Running shutdown script ~~~~~~~"

PID="$(pgrep -o "python")"

curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"
preempted=$?

echo "Got preempted: $preempted"

# if $preempted; then
  # echo "Detected pre-emption"

echo "Send SIGTERM to python PID $PID"
kill "$PID"
echo "SIGTERM sent"

  # Give the python script as much time as possible to cleanup
while true; do echo 'Keep Alive'; sleep 1; done
# fi

echo "~~~~~~~ Completed shutdown script ~~~~~~~"