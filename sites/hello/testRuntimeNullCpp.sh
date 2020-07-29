#!/bin/bash

MAKEFILE=Makefile_runtime_null_cpp
BINARY=test_runtime_null_cpp.x 

TESTDIR=../../test

# Build test binary
if   [[ "$#" -eq 0 ]]; then
        make -f $MAKEFILE clean all
elif [[ "$#" -eq 1 ]]; then
    if [[ "$1" = "--debug" ]]; then
        make -f $MAKEFILE clean all DEBUG=T
    else
        echo "Unknown command line argument", $1
        exit 1;
    fi
elif [[ "$#" -gt 1 ]]; then
    echo "At most one command line argument accepted"
    exit 2;
fi

# Confirm build and clean-up
if [[ $? -ne 0 ]]; then
    echo "Unable to compile $BINARY"
    exit 3;
fi

time ./$BINARY
if [[ $? -ne 0 ]]; then
    echo "Unable to execute $BINARY successfully"
    exit 4;
fi

