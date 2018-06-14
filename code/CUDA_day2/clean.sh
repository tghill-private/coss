#!/bin/bash

# Cleaning up: deleting executables, and files with *~ names

\rm `find -type f -name \*~`

for f in `find -type f`; do A=`file $f |grep "LSB executable"`; if test "$A"; then \rm $f; fi; done
