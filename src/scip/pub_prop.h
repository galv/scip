/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2004 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2004 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the SCIP Academic Licence.        */
/*                                                                           */
/*  You should have received a copy of the SCIP Academic License             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: pub_prop.h,v 1.1 2004/09/23 15:46:31 bzfpfend Exp $"

/**@file   pub_prop.h
 * @brief  public methods for propagators
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __PUB_PROP_H__
#define __PUB_PROP_H__


#include "def.h"
#include "type_misc.h"
#include "type_prop.h"



/** compares two propagators w. r. to their priority */
extern
DECL_SORTPTRCOMP(SCIPpropComp);

/** gets user data of propagator */
extern
PROPDATA* SCIPpropGetData(
   PROP*            prop                /**< propagator */
   );

/** sets user data of propagator; user has to free old data in advance! */
extern
void SCIPpropSetData(
   PROP*            prop,               /**< propagator */
   PROPDATA*        propdata            /**< new propagator user data */
   );

/** gets name of propagator */
extern
const char* SCIPpropGetName(
   PROP*            prop                /**< propagator */
   );

/** gets description of propagator */
extern
const char* SCIPpropGetDesc(
   PROP*            prop                /**< propagator */
   );

/** gets priority of propagator */
extern
int SCIPpropGetPriority(
   PROP*            prop                /**< propagator */
   );

/** gets frequency of propagator */
extern
int SCIPpropGetFreq(
   PROP*            prop                /**< propagator */
   );

/** gets time in seconds used in this propagator */
extern
Real SCIPpropGetTime(
   PROP*            prop                /**< propagator */
   );

/** gets the total number of times, the propagator was called */
extern
Longint SCIPpropGetNCalls(
   PROP*            prop                /**< propagator */
   );

/** gets total number of times, this propagator detected a cutoff */
extern
Longint SCIPpropGetNCutoffs(
   PROP*            prop                /**< propagator */
   );

/** gets total number of domain reductions found by this propagator */
extern
Longint SCIPpropGetNDomredsFound(
   PROP*            prop                /**< propagator */
   );

/** is propagator initialized? */
extern
Bool SCIPpropIsInitialized(
   PROP*            prop                /**< propagator */
   );


#endif
