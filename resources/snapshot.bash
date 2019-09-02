#!/bin/bash

distpath=./datasets/A
mkdir -p $distpath
/usr/bin/raspistill -w 320 -h 240 -t 1000 -o ./$distpath/snap_$(date +%Y%m%d_%H%M%S).jpg
