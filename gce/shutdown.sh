echo "Running shutdown script"
preempted = curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"
if $preempted; then
  PID="$(pgrep -o "python")"
  echo "Send SIGTERM to python PID $PID"
  kill "$PID"
  sleep infinity
fi
echo "Completed shutdown script"