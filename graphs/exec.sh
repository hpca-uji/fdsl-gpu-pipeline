#!/bin/bash

# Activate virtualenv
source $HOME/localenvs/rtfdslenv/bin/activate

python nvertex/table.py
python res/table.py
python batchsize/table.py
python training/table.py