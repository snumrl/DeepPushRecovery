#!/bin/bash
git pull;cd .. ; rm -rf build; mkdir build; cd build; cmake ..; make -j8; cd ../python ;cd pushrecoverybvhgenerator ; git pull;cd /mass/push_exp
