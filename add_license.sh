#!/bin/bash

echo -e "$(bat LICENSE)\n$(bat $1)" > $1
echo "Added the LICENSE statement to file: $1."
echo "Don't forget to comment out the LICENSE!"
