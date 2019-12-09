#!/bin/bash
git pull ; cd .. ; rm -rf build; mkdir build; cd build; cmake ..; make -j8; cd ../python ;cd pushrecoverybvhgenerator ; git pull ; cd ../../push_exp ; python3 main_CrouchSimulation.py "$@"
