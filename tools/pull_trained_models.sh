#!/bin/bash -ex

read -p 'Robosub Server Username: ' username

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

repo_base=$script_directory/../
rsync -Prh $username@robosub.eecs.wsu.edu:/data/vision/trained_models $repo_base
