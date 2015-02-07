#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments supplied: using /rsbopencv/ipl as output scope"
    rsb_videosender -o /rsbopencv/ipl
else
    rsb_videosender -o $1
fi
