#!/bin/bash

echo -e "\"\"\"$(bat LICENSE)\"\"\"\n$(bat $1)" > $1
