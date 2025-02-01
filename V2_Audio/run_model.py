# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a model.

Useage:
python3 run_model.py --model_file model_edgetpu.tflite
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import model
import numpy as np
import pickle

from procesamiento_audio2 import train_data
def main():
  parser = argparse.ArgumentParser()
  model.add_model_flags(parser)
  args = parser.parse_args()
  ############## CARGA DEL MODELO ENTRENADO ########################
  file_path = '../V1_Audio/modelo_normal_largo.pkl'
  with open(file_path, 'rb') as file:
      gm = pickle.load(file)
  gm = gm.fit(train_data)
  interpreter = gm
  mic = args.mic if args.mic is None else int(args.mic)
  model.classify_audio(mic, interpreter,
                       sample_rate_hz=int(args.sample_rate_hz),
                       num_frames_hop=int(args.num_frames_hop))

if __name__ == "__main__":
  main()
