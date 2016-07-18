#!/bin/bash

trace() {
	FILE=$1;
	THRESHOLD=$2;
	SILENCE=$3;
	RENDER=$4;
	echo 'THRESHOLD:' $THRESHOLD
	echo 'SILENCE:' $SILENCE
	echo 'RENDER:' $RENDER
	python3 -c "from rivuletpy.trace import trace; trace('$FILE', threshold=$THRESHOLD, render=$RENDER, length=4, toswcfile='$FILE.rivuet.swc', ignore_radius=True, clean=False, silence=$SILENCE)"
}
export -f trace

# Check for options first
while [[ $# -gt 0 ]]
do
key="$1"

THRESHOLD=0
RENDER=False
SILENCE=True
THREAD=4

case $key in
	-t|--THRESHOLD)
	export THRESHOLD="$2"
	shift # past argument
	;;
	-j|--THREAD)
	export THREAD="$2"
	echo Setting THREAD "$2"
	shift # past argument
	;;
	-s|--SILENCE)
	export SILENCE="$2"
	shift # past argument
	;;
	*)
	echo "Doing $1"
	sem -j$THREAD trace "$1" "$THRESHOLD" "$SILENCE" "$RENDER"; # When it is not an option, it is assumed to be a file
	;;
esac

shift
done
sem --wait

echo "==== ALL done ===="