#!/bin/bash
git pull ; cd pushrecoverybvhgenerator ; git pull ; cd .. ; python3 PPO.py "$@"
