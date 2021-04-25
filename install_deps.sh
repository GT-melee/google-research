# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

python3 -m venv ./venv
source ./venv/bin/activate

pip install tf-nightly==2.5.0.dev20210326
pip install tb-nightly==2.5.0a20210405

git clone https://github.com/openai/gym.git ../gym
pip install -e ../gym

pip3 install -r social_rl/requirements.txt