/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   heur_conflictdiving.c
 * @brief  LP diving heuristic that chooses fixings w.r.t. soft locks
 * @author Jakob Witzig
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/heur_conflictdiving.h"

#define HEUR_NAME                    "conflictdiving"
#define HEUR_DESC                    "LP diving heuristic that chooses fixings w.r.t. soft locks"
#define HEUR_DISPCHAR                '~'
#define HEUR_PRIORITY                -1000250
#define HEUR_FREQ                    1
#define HEUR_FREQOFS                 0
#define HEUR_MAXDEPTH                -1
#define HEUR_TIMING                  SCIP_HEURTIMING_DURINGLPLOOP
#define HEUR_USESSUBSCIP             FALSE  /**< does the heuristic use a secondary SCIP instance? */
#define DIVESET_DIVETYPES            SCIP_DIVETYPE_INTEGRALITY | SCIP_DIVETYPE_SOS1VARIABLE /**< bit mask that represents all supported dive types */
#define DEFAULT_RANDSEED             151 /**< default random seed */

/*
 * Default parameter settings
 */

#define DEFAULT_MINRELDEPTH         0.0 /**< minimal relative depth to start diving */
#define DEFAULT_MAXRELDEPTH         1.0 /**< maximal relative depth to start diving */
#define DEFAULT_MAXLPITERQUOT      0.05 /**< maximal fraction of diving LP iterations compared to node LP iterations */
#define DEFAULT_MAXLPITEROFS       1000 /**< additional number of allowed LP iterations */
#define DEFAULT_MAXDIVEUBQUOT       0.8 /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOT      0.0 /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEUBQUOTNOSOL  0.1 /**< maximal UBQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOTNOSOL 0.0 /**< maximal AVGQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_BACKTRACK          TRUE /**< use one level of backtracking if infeasibility is encountered? */
#define DEFAULT_LPRESOLVEDOMCHGQUOT 0.01 /**< percentage of immediate domain changes during probing to trigger LP resolve */
#define DEFAULT_LPSOLVEFREQ           0 /**< LP solve frequency for diving heuristics */
#define DEFAULT_ONLYLPBRANCHCANDS FALSE /**< should only LP branching candidates be considered instead of the slower but
                                         *   more general constraint handler diving variable selection? */

/* locally defined heuristic data */
struct SCIP_HeurData
{
   SCIP_SOL*             sol;                /**< working solution */
};

/*
 * local methods
 */

/*
 * Callback methods
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
static
SCIP_DECL_HEURCOPY(heurCopyConflictdiving)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* call inclusion method of constraint handler */
   SCIP_CALL( SCIPincludeHeurConflictdiving(scip) );

   return SCIP_OKAY;
}

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeConflictdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);

   /* free heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   SCIPfreeBlockMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitConflictdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* create working solution */
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitConflictdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free working solution */
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );

   return SCIP_OKAY;
}


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecConflictdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;
   SCIP_DIVESET* diveset;

   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   assert(SCIPheurGetNDivesets(heur) > 0);
   assert(SCIPheurGetDivesets(heur) != NULL);
   diveset = SCIPheurGetDivesets(heur)[0];
   assert(diveset != NULL);

   SCIP_CALL( SCIPperformGenericDivingAlgorithm(scip, diveset, heurdata->sol, heur, result, nodeinfeasible) );

   return SCIP_OKAY;
}

/** returns a score for the given candidate -- the best candidate maximizes the diving score */
static
SCIP_DECL_DIVESETGETSCORE(divesetGetScoreConflictdiving)
{
   SCIP_Real softlocksum = SCIPvarGetNLocksSoftDown(cand) + SCIPvarGetNLocksSoftUp(cand);
   SCIP_Bool mayrounddown = SCIPvarMayRoundDown(cand);
   SCIP_Bool mayroundup = SCIPvarMayRoundUp(cand);

   /* variable can be rounded in exactly one direction */
   if( mayrounddown != mayroundup )
   {
      *roundup = mayroundup;
   }
   else
   {
      *roundup = (SCIPvarGetNLocksSoftDown(cand) >= SCIPvarGetNLocksSoftUp(cand));
   }

   if( *roundup )
   {
      switch( divetype )
      {
         case SCIP_DIVETYPE_INTEGRALITY:
            candsfrac = 1.0 - candsfrac;
            break;
         case SCIP_DIVETYPE_SOS1VARIABLE:
            if ( SCIPisFeasPositive(scip, candsol) )
               candsfrac = 1.0 - candsfrac;
            break;
         default:
            SCIPerrorMessage("Error: Unsupported diving type\n");
            SCIPABORT();
            return SCIP_INVALIDDATA; /*lint !e527*/
      } /*lint !e788*/
      *score = SCIPvarGetNLocksUp(cand)/MAX(1.0, softlocksum) + 0.0001 * SCIPvarGetNLocksSoftUp(cand);
   }
   else
   {
      if ( divetype == SCIP_DIVETYPE_SOS1VARIABLE && SCIPisFeasNegative(scip, candsol) )
         candsfrac = 1.0 - candsfrac;
      *score = SCIPvarGetNLocksSoftDown(cand)/MAX(1.0, softlocksum) + 0.0001 * SCIPvarGetNLocksDown(cand);
   }


   /* penalize too small fractions */
   if( candsfrac < 0.01 )
      (*score) *= 0.1;

   /* prefer decisions on binary variables */
   if( !SCIPvarIsBinary(cand) )
      (*score) *= 0.1;

   /* penalize the variable if it may be rounded. */
   if( mayrounddown || mayroundup )
      (*score) -= SCIPgetNLPRows(scip);

   /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
   assert( (0.0 < candsfrac && candsfrac < 1.0) || SCIPvarIsBinary(cand) || divetype == SCIP_DIVETYPE_SOS1VARIABLE );

   return SCIP_OKAY;
}

/*
 * heuristic specific interface methods
 */

/** creates the conflictdiving heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurConflictdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_HEUR* heur;

   /* create conflictdiving primal heuristic data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ,
         HEUR_FREQOFS, HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecConflictdiving, heurdata) );

   assert(heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurCopy(scip, heur, heurCopyConflictdiving) );
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeConflictdiving) );
   SCIP_CALL( SCIPsetHeurInit(scip, heur, heurInitConflictdiving) );
   SCIP_CALL( SCIPsetHeurExit(scip, heur, heurExitConflictdiving) );

   /* create a diveset (this will automatically install some additional parameters for the heuristic)*/
   SCIP_CALL( SCIPcreateDiveset(scip, NULL, heur, HEUR_NAME, DEFAULT_MINRELDEPTH, DEFAULT_MAXRELDEPTH, DEFAULT_MAXLPITERQUOT,
         DEFAULT_MAXDIVEUBQUOT, DEFAULT_MAXDIVEAVGQUOT, DEFAULT_MAXDIVEUBQUOTNOSOL, DEFAULT_MAXDIVEAVGQUOTNOSOL, DEFAULT_LPRESOLVEDOMCHGQUOT,
         DEFAULT_LPSOLVEFREQ, DEFAULT_MAXLPITEROFS, DEFAULT_RANDSEED, DEFAULT_BACKTRACK, DEFAULT_ONLYLPBRANCHCANDS, DIVESET_DIVETYPES, divesetGetScoreConflictdiving) );

   return SCIP_OKAY;
}

