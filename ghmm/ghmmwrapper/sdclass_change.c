/*******************************************************************************
*
*       This file is part of the General Hidden Markov Model Library,
*       GHMM version __VERSION__, see http://ghmm.org
*
*       Filename: sdclass_change.c
*       Authors:  Benjamin Georgi
*
*       Copyright (C) 1998-2004 Alexander Schliep
*       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
*       Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
*                               Berlin
*
*       Contact: schliep@ghmm.org
*
*       This library is free software; you can redistribute it and/or
*       modify it under the terms of the GNU Library General Public
*       License as published by the Free Software Foundation; either
*       version 2 of the License, or (at your option) any later version.
*
*       This library is distributed in the hope that it will be useful,
*       but WITHOUT ANY WARRANTY; without even the implied warranty of
*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*       Library General Public License for more details.
*
*       You should have received a copy of the GNU Library General Public
*       License along with this library; if not, write to the Free
*       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*
*
*       This file is version $Revision: 1451 $
*                       from $Date: 2005-10-18 13:21:55 +0300 (Tue, 18 Oct 2005) $
*             last change by $Author: grunau $.
*
*******************************************************************************/

/* XXX FIXME: breaks out of tree build of ghmmwrapper */
#include "../config.h"

#include <stdio.h>
#include <stdlib.h>
#include <ghmm/rng.h>
#include <ghmm/sequence.h>
#include <ghmm/sdmodel.h>

int cp_class_change(int *seq, int len) {
  int sum = 0;
  int i;
  for(i=0;i<=len;i++){
	  sum += seq[i];
  }
  //printf("sum = %d\n",sum);
  if (sum >= 6) {
    //printf("\n++++++++++++++++++++++++++++++++Switching class .... ");    
    return 1;
  }
  else {
    return 0;
  } 
} 		
		

void setSwitchingFunction( ghmm_dsmodel *smd ) {
  smd->get_class = cp_class_change;
}

