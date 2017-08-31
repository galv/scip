#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

TSTNAME=$1
BINNAME=$2
SETNAMES=$3
BINID=$4
TIMELIMIT=$5
NODELIMIT=$6
MEMLIMIT=$7
THREADS=$8
FEASTOL=$9
DISPFREQ=${10}
CONTINUE=${11}
LOCK=${12}
VERSION=${13}
LPS=${14}
DEBUGTOOL=${15}
CLIENTTMPDIR=${16}
REOPT=${17}
OPTCOMMAND=${18}
SETCUTOFF=${19}
MAXJOBS=${20}
VISUALIZE=${21}
PERMUTE=${22}
SEEDS=${23}

SOLVER=fscip

# check if all variables defined (by checking the last one)
if test -z $SEEDS
then
    echo Skipping test since not all variables are defined
    echo "TSTNAME       = $TSTNAME"
    echo "BINNAME       = $BINNAME"
    echo "SETNAMES      = $SETNAMES"
    echo "BINID         = $BINID"
    echo "TIMELIMIT     = $TIMELIMIT"
    echo "NODELIMIT     = $NODELIMIT"
    echo "MEMLIMIT      = $MEMLIMIT"
    echo "THREADS       = $THREADS"
    echo "FEASTOL       = $FEASTOL"
    echo "DISPFREQ      = $DISPFREQ"
    echo "CONTINUE      = $CONTINUE"
    echo "LOCK          = $LOCK"
    echo "VERSION       = $VERSION"
    echo "LPS           = $LPS"
    echo "DEBUGTOOL      = $DEBUGTOOL"
    echo "CLIENTTMPDIR  = $CLIENTTMPDIR"
    echo "REOPT         = $REOPT"
    echo "PERMUTE       = $PERMUTE"
    echo "OPTCOMMAND    = $OPTCOMMAND"
    echo "SETCUTOFF     = $SETCUTOFF"
    echo "MAXJOBS       = $MAXJOBS"
    echo "VISUALIZE     = $VISUALIZE"
    echo "PERMUTE       = $PERMUTE"
    echo "SEEDS         = $SEEDS"
    exit 1;
fi

# call routines for creating the result directory, checking for existence
# of passed settings, etc
TIMEFORMAT="sec"
MEMFORMAT="kB"
. ./configuration_set_fscip.sh $BINNAME $TSTNAME $SETNAMES $TIMELIMIT $TIMEFORMAT $MEMLIMIT $MEMFORMAT $DEBUGTOOL $SETCUTOFF

EXECNAME=$SCIPPATH/../bin/$BINNAME
echo $EXECNAME

# check if we can set hard memory limit (address, leak, or thread sanitzer don't like ulimit -v)
if [ `uname` == Linux ] && (ldd ${EXECNAME} | grep -q lib[alt]san) ; then
   # skip hard mem limit if using AddressSanitizer (libasan), LeakSanitizer (liblsan), or ThreadSanitizer (libtsan)
   HARDMEMLIMIT="none"
elif [ `uname` == Linux ] && (nm ${EXECNAME} | grep -q __[alt]san) ; then
   # skip hard mem limit if using AddressSanitizer, LeakSanitizer, or ThreadSanitizer linked statitically (__[alt]san symbols)
   HARDMEMLIMIT="none"
else
   ULIMITMEM="ulimit -v $HARDMEMLIMIT k;"
fi

export EXECNAME=${DEBUGTOOLCMD}${EXECNAME}

INIT="true"
COUNT=0
for idx in ${!INSTANCELIST[@]}
do
    # retrieve instance and timelimits from arrays set in the configuration_set.sh script
    INSTANCE=${INSTANCELIST[$idx]}
    TIMELIMIT=${TIMELIMLIST[$idx]}
    HARDTIMELIMIT=${HARDTIMELIMLIST[$idx]}

    COUNT=`expr $COUNT + 1`

    # run different random seeds
    for ((s = 0; $s <= $SEEDS; s++))
    do

	# permute transformed problem
	for ((p = 0; $p <= $PERMUTE; p++))
	do

	    # loop over settings
	    for SETNAME in ${SETTINGSLIST[@]}
	    do
		# waiting while the number of jobs has reached the maximum
		if [ $MAXJOBS -ne 1 ]
		then
			while [ `jobs -r|wc -l` -ge $MAXJOBS ]
			do
			    sleep 10
			    echo "Waiting for jobs to finish."
			done
		fi

	    # infer the names of all involved files from the arguments
		QUEUE=`hostname`


		# infer the names of all involved files from the arguments
		. ./configuration_logfiles_fscip.sh $INIT $COUNT $INSTANCE $BINID $PERMUTE $SEEDS $SETNAME $TSTNAME $CONTINUE $QUEUE $THREADS $p $s

		if test "$INSTANCE" = "DONE"
		then
		    wait
		    #echo $EVALFILE
		    ./evalcheck_cluster.sh $EVALFILE
		    continue
		fi

		if test "$SKIPINSTANCE" = "true"
		then
		    continue
		fi

		# find out the solver that should be used
		SOLVER=`stripversion $BINNAME`

		# additional environment variables needed by run.sh
		export SOLVERPATH=$SCIPPATH
		export BASENAME=$FILENAME
		export FILENAME=$INSTANCE
		# export SOLNAME=$SOLCHECKFILE
                export SETNAME
                export THREADS
                export TIMELIMIT
		export CLIENTTMPDIR
		export CHECKERPATH=$SCIPPATH/solchecker

		echo Solving instance $INSTANCE with settings $SETNAME, hard time $HARDTIMELIMIT, hard mem $HARDMEMLIMIT
		if [ $MAXJOBS -eq 1 ]
		then
		    bash -c "ulimit -t $HARDTIMELIMIT s; $ULIMITMEM ulimit -f 200000; ./run_fscip.sh"
		else
		    bash -c "ulimit -t $HARDTIMELIMIT s; $ULIMITMEM ulimit -f 200000; ./run_fscip.sh" &
		fi
		#./run.sh
	    done # end for SETNAME
	done # end for PERMUTE
    done # end for SEEDS

    # after the first termination of the set loop, no file needs to be initialized anymore
    INIT="false"
done # end for TSTNAME