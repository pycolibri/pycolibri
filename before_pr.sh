#!/bin/bash
rm -r docs/build docs/source/auto_examples docs/source/gen_modules docs/source/stubs docs/source/sg_execution_times.rst
make html -C docs/
pytest -v -s
open docs/build/html/index.html 
