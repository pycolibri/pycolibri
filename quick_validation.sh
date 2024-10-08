#!/bin/bash
if [[ "$1" == "--docs" ]]; then
    rm -r docs/build docs/source/auto_examples docs/source/gen_modules docs/source/stubs docs/source/sg_execution_times.rst
    make html -C docs/
    open docs/build/html/index.html
elif [[ "$1" == "--tests" ]]; then
    pytest -v -s
else
    rm -r docs/build docs/source/auto_examples docs/source/gen_modules docs/source/stubs docs/source/sg_execution_times.rst
    make html -C docs/
    open docs/build/html/index.html
    pytest -v -s
fi
