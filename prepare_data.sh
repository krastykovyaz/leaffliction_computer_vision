#!/bin/bash

unzip leaves.zip
mkdir Apple/
mv images/Grape_Black_rot/ Apple/apple_black_rot
mv images/Apple_healthy/ Apple/apple_healthy
mv images/Apple_rust/ Apple/apple_rust
mv images/Apple_scab/ Apple/apple_scab
rm -rf images
