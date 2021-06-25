#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
   open neural_mmo/forge/embyr/UnityClient/neural-mmo.app
else
   ./neural_mmo/forge/embyr/UnityClient/neural-mmo.x86_64
fi
