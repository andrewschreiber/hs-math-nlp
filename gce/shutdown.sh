
# Re-add !/bin/bash if using outside of latest-pytorch-gpu
# If this doesn't work, search for your image using set-metadata

echo "~~~~~~~ Running shutdown script ~~~~~~~"

OLD_PID=$(pgrep -o "python")

# curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google"
# preempted=$?

echo "Send SIGTERM to python OLD_PID $OLD_PID"
kill $OLD_PID
echo "SIGTERM sent"

# Give the python script as much time as possible to cleanup
# Actually this doesn't matter. Google instakills / corrupts preemptible GPU instances. You must save periodically, ahead of time.
# for i in {1..30}; do echo 'Keep Alive'; sleep 1; done

echo "~~~~~~~ Completed shutdown script ~~~~~~~"