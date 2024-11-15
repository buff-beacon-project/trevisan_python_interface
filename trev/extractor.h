/* This file is part of libtrevisan, a modular implementation of
   Trevisan's randomness extraction construction.

   Copyright (C) 2011-2012, Wolfgang Mauerer <wm@linux-kernel.net>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with libtrevisan. If not, see <http://www.gnu.org/licenses/>. */

#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include<string>
#include<fstream>
#include<iostream>
#include "primitives.h"
#include "phys_params.h"
//#include "R_interp.h"				//ROMMAL EDIT 8-5-13, commented out include statement including R-code

// All parameters that are required to describe the properties
// of a Trevisan extractor
struct params {
	struct phys_params pp;

	wd_type wdt;	   // Weak design
	bext_type bxt;	   // Bit extractor
	//wd_type basic_wdt; // Basic design for the block design		//ROMMAL EDIT 7-30-13, commented out block
																	//design variable, unused in this version
	int num_tasks; // 0 means unlimited
     	int num_threads;    // Mink 6-27-2018, num of PThreads, 0 means none

	bool verbose;	     // Enable chit-chat?
	bool dryrun;	     // Perform only parameter calculations
	bool ignore_entropy; // Ignore that extractor requires a larger k than available?
 	int  si_tst ;        // Mink 9-10-2014, add opt Si_test
     	bool blk_des; 	     // Mink 9-10-2014, add opt Block Design
     	bool streaming;	     // Mink 7-20-2016, add opt for QRNG proj data streamin

	bool skip_bitext;      // Skip the bit-extraction step?
	bool save_weakdes; // Save the weak design?
	// std::string wd_filename;				//ROMMAL EDIT 7-30-13, commented out unused variables
	// std::ofstream wd_out;
	
	//R_interp *R;							//ROMMAL EDIT 8-5-13, commented out include statement including R-code
};

int dispatch(struct params &params);

#endif
