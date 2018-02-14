#!/bin/bash -e

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

repo_base=`dirname $script_directory`
cd $repo_base
git submodule init
git submodule update

python_path_ext="$repo_base/models/research:$repo_base/models/research/slim"

echo 'export PYTHONPATH=$PYTHONPATH:'$python_path_ext >> ~/.bashrc

cd $repo_base/models/research && protoc object_detection/protos/*.proto --python_out=.
