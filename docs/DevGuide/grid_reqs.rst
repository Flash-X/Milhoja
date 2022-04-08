Requirements
============

1. At configuration, calling code shall provide Milhoja's grid backend with a
routine that Milhoja can use on a per-block-basis to determine which cells in
the block request refinement of the block, which request the same refinement,
and which indicate that derefinement of the block is acceptable.  The final
determination of whether to refine, derefine, or do nothing shall be determined
solely by the grid backend.  This routine is referred to as the refinement error
estimation routine.  Note that for simplicity's sake, this routine shall only be
called with blocks rather than proper tiles.

2. Calling code could desire to have a block refined in the case that the data
in a block's interior does not indicate that this is necessary but the data in
the block's guardcells do.  This could be a good idea, for instance, if a shock
in the guardcells were expected to propagate into the interior cells before the
next regridding.  Therefore, the refinement error estimation routine shall be
provided with information to allow if to only estimate error based on the
interior cells or to estimate error on the guardcells in the same manner as the
interior cells.  This implies that the routine will be given lo/hi for the
interior as well as for the block with GCs.

3. Calling code could desire to estimate refinement error for a particular cell
based on the data of several variables.  Therefore, the refinement error
estimation routine shall be provided with all current cell-centered data for the
given block including for guardcells.

4. At configuration, calling code shall provide Milhoja's grid backend with a
routine that Milhoja can use on a per-tile basis to set cell-centered initial
conditions in the interior cells of tiles that are created during the process of
establishing the initial AMR refinement.  As part of this process, the grid
backend might use the refinement error estimation routine to assess
refinement/derefinement decisions and, therefore, this initial conditions
routine must write at the very least all data used by the refinement error
estimation routine.

5. Since the refinement error estimation routine can be used during
initialization, requirement XXX implies that the grid backend is reponsible for
filling the guardcells of all cell-centered variables (it does not know which
variables were set) as well as the boundary conditions.  Clearly it shall
perform this work after using the IC routine but before refinement errors are
estimated.  NOTE: A more sensible requirement would be that the IC routine set
data in the interiors and GCs.  Then the grid backend would only need to
overwrite GCs outside of the domain with BCs.  However, Flash-X is not
compatible with this and Klaus thinks that some Simulation_initBlock routines
that handle BCs in a special way would be very difficult to rewrite for this
alternate req.

QUESTION: Should errors estimated in GCs that contain BC data be
used in determining refinement?  Is it possible that they contain BC data that
indicates refinement even when the data in the neighboring interior cells does
not require it?  What if not flagging BC data influences the refinement decision
(black box) such that a block is *not* refined despite the interior data needing
refinement?  No information on this in the Flash4 user guide.  Klaus confirmed
that we should do some refinement based on GC contents.  Need to study
gr_estimateBlkError carefully to get the final answer.

6. Calling code shall configure the grid backend with the interpolator to be
used when interpolating fine GC data from coarse data at a fine/coarse boundary.

