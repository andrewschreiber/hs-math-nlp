echo "~~~~~~~ Running shutdown script ~~~~~~~"

preempted = curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"

if $preempted; then
  print("Detected pre-emption")
  PID="$(pgrep -o "python")"

  echo "Send SIGTERM to python PID $PID"
  kill "$PID"
  echo "SIGTERM sent"

  sleep infinity
fi

echo "~~~~~~~ Completed shutdown script ~~~~~~~"