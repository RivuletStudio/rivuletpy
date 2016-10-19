#!/bin/bash

../rivuletenv/bin/python3 setup.py clean;
../rivuletenv/bin/python3 setup.py build;
../rivuletenv/bin/pip3 install . --upgrade
../rivuletenv/bin/python3 tests/testmsfm.py
