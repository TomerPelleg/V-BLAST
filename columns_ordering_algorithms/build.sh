#!/usr/bin/env bash

set -e

echo "Cleaning previous build..."
rm -rf build

mkdir build
cd build
cmake ..
make