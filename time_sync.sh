#!/bin/bash
# This script syncs Jetson time with NUC time. This is a hacky way,
# need to make a better long term solution using NTP or chrony

# prompt for login information
echo -n "user: "
read USER

sudo date --set="$(ssh ${USER}@cobalt date)"
