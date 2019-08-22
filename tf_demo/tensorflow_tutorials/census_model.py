import os
import sys

models_path = os.path.join(os.getcwd(), "models")
sys.path.append(models_path)

from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data")
