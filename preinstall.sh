#!/bin/bash
pip install --upgrade pip
pip install numpy==1.24.4  # Explicit old version
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
