#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--name) name="$2"; shift ;;
        -s|--sleeps) sleeps="$3"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

gcloud compute connect-to-serial-port $name --port 2

sleep 20

./watch_ttyS2.sh $name
