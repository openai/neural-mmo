#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
   open forge/embyr/UnityClient/neural-mmo.app
else
   ./forge/embyr/UnityClient/neural-mmo.x86_64
fi
