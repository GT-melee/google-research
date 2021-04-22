#!/bin/bash
set -e
set -x

python3 -m venv ./venv
source ./venv/bin/activate

pip3 install tf-nightly==2.5.0.dev20210326
pip3 install tb-nightly==2.5.0a20210405

pip3 install -r social_rl/requirements.txt
