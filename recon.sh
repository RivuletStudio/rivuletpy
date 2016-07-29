#!/bin/bash

trace() {
	FILE=$1;
	THRESHOLD=$2;
	FILTER=$3;
	RADII=$4;
	FILTERTHRESHOLD=$5;
	echo 'THRESHOLD:' $THRESHOLD
	echo 'SILENCE:' $SILENCE
	echo 'RENDER:' $RENDER
	python3 -c "from rivuletpy.recon import recon; import numpy as np; recon('$FILE', $THRESHOLD, $FILTERTHRESHOLD, filter='$FILTER', radii=np.arange($RADII))"
}

export -f trace

# Check for options first
while [[ $# -gt 0 ]]
do
key="$1"

THRESHOLD=0
FILTERTHRESHOLD=4
RADII="2.6,3,0.1"
FILTER="bg"
THREAD=1

case $key in
	-t|--THRESHOLD)
	export THRESHOLD="$2"
	shift # past argument
	;;
	-l|--FILTERTHRESHOLD)
	export THRESHOLD="$2"
	shift # past argument
	;;
	-j|--THREAD)
	export THREAD="$2"
	echo Setting THREAD "$2"
	shift # past argument
	;;
	-r|--RADII)
	export RADII="$2"
	shift # past argument
	;;
	-f|--FILTER)
	export FILTER="$2"
	shift # past argument
	;;
	*)
	echo "Doing $1"
	# sem -j$THREAD trace "$1" "$THRESHOLD" "$FILTER" "$RADII" ; # When it is not an option, it is assumed to be a file
	trace "$1" "$THRESHOLD" "$FILTER" "$RADII" "$FILTERTHRESHOLD";
	;;
esac

shift
done
# sem --wait

echo "==== ALL done ===="