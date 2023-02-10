#!/bin/bash
pip install -r requirements.txt
ipython Aiyagari-HARK.ipynb
jupyter nbconvert --to python Aiyagari-HARK.ipynb 
#Successful nbconverting might require updating 'mistune' and 'nbconvert' and rebooting your environment.