import subprocess
import csv
import tempfile
import os
import numpy as np
import time
import argparse
import errno
import copy
from lxml import etree
import logging
import pdb

PARAMETER_LIMITS = {
    "X": (0, 4),
    "Y": (0, 1),
    "Z": (0, 4)
}

parameters = {}


for key in PARAMETER_LIMITS:
        if key == "X":
            parameters[key] = str(np.random.exponential(0.1))
        elif key == "RateOfInformedSearchDecay":
            parameters[key] = str(np.random.exponential(0.2))
        else:
            parameters[key] = str(np.random.uniform(PARAMETER_LIMITS[key][0], PARAMETER_LIMITS[key][1]))


# run
def run_generation(self):
    
