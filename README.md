# About this fork

I forked the original repo to upgrade its dependencies (the most disruptive one being from Tensorflow 1.0 to Tensorflow 2.0).
To test the newer dependencies, do NOT run `python setup.py install`. This will install scPhere as a python package and would make 
refactoring the code of the repo more challenging. Instead, run the test script `example/script/demo.py` and install the dependencies manually (preferably in a clean conda environment). The script was modified to use the scPhere package through relative imports.

# About the project

See READ.me of original repo for more information.
