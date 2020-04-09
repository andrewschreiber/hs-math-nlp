#!/bin/bash

echo "++++++++++++++ Shutdown test +++++++++++++++++"

ofile=/var/tmp/shutdown.txt 
echo "id = $(id)" > $ofile 
echo "script_file path = $(realpath $0)" >> $ofile 
echo "script_file rights, user, group = $(stat -c "%A %U %G" $0)" >> $ofile

sleep 30

echo "post sleep"

exit 0