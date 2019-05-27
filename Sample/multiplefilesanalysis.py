import os
from glob import glob
cwd = os.getcwd() #current working directory
filenames = glob(cwd + "/results/*") #folder in which it is supposed to be stored the accelerometers data files
