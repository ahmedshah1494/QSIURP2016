/*******************************************************************************
*
*       This file is part of the General Hidden Markov Model Library,
*       GHMM version __VERSION__, see http://ghmm.org
*
*       Filename: ghmm/ghmm/ghmmconfig.h.in
*       Authors:  Janne Grunau
*
*       Copyright (C) 1998-2004 Alexander Schliep 
*       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
*	Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik, 
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
*                       from $Date: 2005-10-18 12:21:55 +0200 (Tue, 18 Oct 2005) $
*             last change by $Author: grunau $.
*
*******************************************************************************/

#ifndef GHMMCONFIG_H
#define GHMMCONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#if 0                 /* GHMM_OBSOLETE */
#define GHMM_OBSOLETE
#endif

#if 0              /* GHMM_UNSUPPORTED */
#define GHMM_UNSUPPORTED
#endif


/* defining the used RNG */

#if 1     /* GHMM_RNG_MERSENNE_TWISTER */
#define GHMM_RNG_MERSENNE_TWISTER
#endif

#if 0                  /* GHMM_RNG_BSD */
#define GHMM_RNG_BSD
#endif

#if 0                  /* GHMM_RNG_GSL */
#define GHMM_RNG_GSL
#endif

#ifdef __cplusplus
}
#endif

#endif /* GHMMCONFIG_H*/
