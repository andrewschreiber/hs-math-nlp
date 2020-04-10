#!/bin/bash

# echo "++++++++++++++ Shutdown test +++++++++++++++++"

ofile=/var/tmp/shutdown3.txt 
echo "++++++++++++++ Shutdown test +++++++++++++++++"
echo "id = $(id)" > $ofile 
echo "script_file path = $(realpath $0)" >> $ofile 
echo "script_file rights, user, group = $(stat -c "%A %U %G" $0)" >> $ofile

while true; do echo 'sec'; sleep 1; done

exit 0

