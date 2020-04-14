
# Readd !/bin/bash if using outside of latest-pytorch-gpu
# If this doesn't work, make sure your image isn't editing your metadata

echo "~~~~~~~ Running shutdown script ~~~~~~~"

OLD_PID=$(pgrep -o "python")
# NEW_PID=$(pgrep -n "python")

# curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"
# preempted=$?

# echo "Got preempted: $preempted"

# if $preempted; then
  # echo "Detected pre-emption"

echo "Send SIGTERM to python OLD_PID $OLD_PID"
kill $OLD_PID
# if [ "$OLD_PID" != "$NEW_PID" ]; then
#   echo "Send SIGTERM to python NEW_PID $NEW_PID"
#   kill $NEW_PID
# fi

echo "SIGTERM sent"

# Give the python script as much time as possible to cleanup
for i in {1..30}; do echo 'Keep Alive'; sleep 1; done

echo "~~~~~~~ Completed shutdown script ~~~~~~~"