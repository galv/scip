#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2015 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
#
# The queue is passed via $QUEUE (possibly defined in a local makefile in scip/make/local).
#
# For each run, we can specify the number of nodes reserved for a run via $PPN. If tests runs
# with valid time measurements should be executed, this number should be chosen in such a way
# that a job is run on a single computer, i.e., in general, $PPN should equal the number of cores
# of each computer. If course, the value depends on the specific computer/queue.
#
# To get the result files call "./evalcheck_cluster_xpress.sh
# results/check.$TSTNAME.$BINNAME.$SETNAME.eval in directory check/
# This leads to result files
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.out
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.res
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.err

TSTNAME=$1
BINNAME=$2
SETNAME=$3
BINID=$BINNAME.$4
TIMELIMIT=$5
NODELIMIT=$6
MEMLIMIT=$7
THREADS=$8
FEASTOL=$9
DISPFREQ=${10}
CONTINUE=${11}
QUEUETYPE=${12}
QUEUE=${13}
PPN=${14}
CLIENTTMPDIR=${15}
NOWAITCLUSTER=${16}
EXCLUSIVE=${17}

# check if all variables defined (by checking the last one)
if test -z $EXCLUSIVE
then
    echo Skipping test since not all variables are defined
    echo "TSTNAME       = $TSTNAME"
    echo "BINNAME       = $BINNAME"
    echo "SETNAMES      = $SETNAME"
    echo "BINID         = $BINID"
    echo "TIMELIMIT     = $TIMELIMIT"
    echo "NODELIMIT     = $NODELIMIT"
    echo "MEMLIMIT      = $MEMLIMIT"
    echo "THREADS       = $THREADS"
    echo "FEASTOL       = $FEASTOL"
    echo "DISPFREQ      = $DISPFREQ"
    echo "CONTINUE      = $CONTINUE"
    echo "QUEUETYPE     = $QUEUETYPE"
    echo "QUEUE         = $QUEUE"
    echo "PPN           = $PPN"
    echo "CLIENTTMPDIR  = $CLIENTTMPDIR"
    echo "NOWAITCLUSTER = $NOWAITCLUSTER"
    echo "EXCLUSIVE     = $EXCLUSIVE"
    exit 1;
fi

# get current SCIP path
SCIPPATH=`pwd`

if test ! -e $SCIPPATH/results
then
    mkdir $SCIPPATH/results
fi

SETTINGS=$SCIPPATH/../settings/$SETNAME.set

# check if the settings file exists
if test $SETNAME != "default"
then
    if test ! -e $SETTINGS
    then
        echo skipping test due to non-existence of settings file $SETTINGS
        exit
    fi
fi

# check if queue has been defined
if test "$QUEUE" = ""
then
    echo Skipping test since the queue name has not been defined.
    exit
fi

# check if number of nodes has been defined
if test "$PPN" = ""
then
    echo Skipping test since the number of nodes has not been defined.
    exit
fi

# check if the slurm blades should be used exclusively
if test "$EXCLUSIVE" = "true"
then
    EXCLUSIVE="  --exclusive "
else
    EXCLUSIVE=""
fi

# we add 100% to the hard time limit and additional 600 seconds in case of small time limits
# NOTE: the jobs should have a hard running time of more than 5 minutes; if not so, these
#       jobs get automatically assigned in the "express" queue; this queue has only 4 CPUs
#       available
HARDTIMELIMIT=`expr \`expr $TIMELIMIT + 600\` + $TIMELIMIT`

#define clusterqueue, which might not be the QUEUE, cause this might be an alias for a bunch of QUEUEs
CLUSTERQUEUE=$QUEUE

NICE=""
ACCOUNT="mip"

if test $CLUSTERQUEUE = "dbg"
then
    CLUSTERQUEUE="mip-dbg,telecom-dbg"
    ACCOUNT="mip-dbg"
elif test $CLUSTERQUEUE = "telecom-dbg"
then
    ACCOUNT="mip-dbg"
elif test $CLUSTERQUEUE = "mip-dbg"
then
    ACCOUNT="mip-dbg"
elif test $CLUSTERQUEUE = "opt-low"
then
    CLUSTERQUEUE="opt"
    NICE="--nice=10000"
fi

# we add 10% to the hard memory limit and additional 100mb to the hard memory limit
HARDMEMLIMIT=`expr \`expr $MEMLIMIT + 100\` + \`expr $MEMLIMIT / 10\``

# in case of qsub queue the memory is measured in kB and in case of srun the time needs to be formatted
if test  "$QUEUETYPE" = "qsub"
then
    HARDMEMLIMIT=`expr $HARDMEMLIMIT \* 1024000`
else
    MYMINUTES=0
    MYHOURS=0
    MYDAYS=0
    #calculate seconds, minutes, hours and days
    MYSECONDS=`expr $HARDTIMELIMIT % 60`
    TMP=`expr $HARDTIMELIMIT / 60`
    if test "$TMP" != "0"
    then
	MYMINUTES=`expr $TMP % 60`
	TMP=`expr $TMP / 60`
	if test "$TMP" != "0"
	then
	    MYHOURS=`expr $TMP % 24`
	    MYDAYS=`expr $TMP / 24`
	fi
   fi
    #format seconds to have two characters
    if test ${MYSECONDS} -lt 10
    then
	MYSECONDS=0${MYSECONDS}
    fi
    #format minutes to have two characters
    if test ${MYMINUTES} -lt 10
    then
	MYMINUTES=0${MYMINUTES}
    fi
    #format hours to have two characters
    if test ${MYHOURS} -lt 10
    then
	MYHOURS=0${MYHOURS}
    fi
    #format HARDTIMELIMT
    if test ${MYDAYS} = "0"
    then
	HARDTIMELIMIT=${MYHOURS}:${MYMINUTES}:${MYSECONDS}
    else
	HARDTIMELIMIT=${MYDAYS}-${MYHOURS}:${MYMINUTES}:${MYSECONDS}
    fi
fi

EVALFILE=$SCIPPATH/results/check.$TSTNAME.$BINID.$QUEUE.$SETNAME"_"$THREADS.eval
echo > $EVALFILE

# counter to define file names for a test set uniquely
COUNT=0

for i in `cat testset/$TSTNAME.test` DONE
do
  if test "$i" = "DONE"
  then
      break
  fi

  # increase the index for the inctance tried to solve, even if the filename does not exist
  COUNT=`expr $COUNT + 1`

  # check if problem instance exists
  if test -f $SCIPPATH/$i
  then

      echo adding instance $COUNT to queue

      # the cluster queue has an upper bound of 2000 jobs; if this limit is
      # reached the submitted jobs are dumped; to avoid that we check the total
      # load of the cluster and wait until it is save (total load not more than
      # 1900 jobs) to submit the next job.
      if test "$NOWAITCLUSTER" != "1"
      then
	  ./waitcluster.sh 1500 $QUEUE 200
      fi

      SHORTFILENAME=`basename $i .gz`
      SHORTFILENAME=`basename $SHORTFILENAME .mps`
      SHORTFILENAME=`basename $SHORTFILENAME .lp`
      SHORTFILENAME=`basename $SHORTFILENAME .opb`

      FILENAME=$USER.$TSTNAME.$COUNT"_"$SHORTFILENAME.$QUEUE.$BINID.$SETNAME"_"$THREADS
      BASENAME=$SCIPPATH/results/$FILENAME

      TMPFILE=$BASENAME.tmp
      SETFILE=$BASENAME.set

      echo $BASENAME >> $EVALFILE

      # in case we want to continue we check if the job was already performed
      if test "$CONTINUE" != "false"
      then
	  if test -e results/$FILENAME.out
	  then
              echo skipping file $i due to existing output file $FILENAME.out
	      continue
	  fi
      fi

      if test -e $SETFILE
      then
	       rm -f $SETFILE
      fi

      echo > $TMPFILE
      echo ""                              > $TMPFILE
      if test $SETNAME != "default"
      then
          echo "non-default settings not yet supported"
      fi
      echo maxtime = -$TIMELIMIT              >> $TMPFILE
      # use wallclock time
      echo cputime = 0                        >> $TMPFILE
      echo miprelstop = $MIPGAP               >> $TMPFILE
      if test $FEASTOL != "default"
      then
          echo miptol = $FEASTOL              >> $TMPFILE
      fi
      echo maxnode = $NODELIMIT               >> $TMPFILE
      echo threads = $THREADS                 >> $TMPFILE

      # the following should enforce the memory limit, but still it is only
      # a soft limit and a temporary file is also written
      echo treememorylimit = $MEMLIMIT        >> $TMPFILE
      echo treememorysavingtarget = 0.0       >> $TMPFILE

      echo format \"timelimit %g\" \$maxtime  >> $TMPFILE
      echo format \"mipgap %g\" \$miprelstop  >> $TMPFILE
      echo format \"feastol %g\" \$miptol     >> $TMPFILE
      echo format \"nodelimit %g\" \$maxnode  >> $TMPFILE
      echo format \"memlimit %g\" \$treememorylimit >> $TMPFILE
      echo format \"percentmemtofile %g\" \$treememorysavingtarget >> $TMPFILE

      echo crossover = 0                      >> $TMPFILE

      echo readprob $SCIPPATH/$i              >> $TMPFILE
      echo time [lpoptimize -b]               >> $TMPFILE
      echo echo bariter:                      >> $TMPFILE
      echo bariter                            >> $TMPFILE
      echo quit                               >> $TMPFILE

      # additional environment variables needed by run.sh
      export SOLVERPATH=$SCIPPATH
      export EXECNAME=$BINNAME
      export BASENAME=$FILENAME
      export FILENAME=$i
      export CLIENTTMPDIR=$CLIENTTMPDIR

      # check queue type
      if test  "$QUEUETYPE" = "srun"
      then
         sbatch --job-name=XPR$SHORTFILENAME --mem=$HARDMEMLIMIT -p $CLUSTERQUEUE -A $ACCOUNT --time=${HARDTIMELIMIT} ${NICE} ${EXCLUSIVE} --output=/dev/null run.sh
      else
         qsub -l walltime=$HARDTIMELIMIT -l mem=$HARDMEMLIMIT -l nodes=1:ppn=$PPN -N XPR$SHORTFILENAME -V -q $QUEUE -o /dev/null -e /dev/null run.sh
      fi
  else
      echo "input file "$SCIPPATH/$i" not found!"
  fi
done