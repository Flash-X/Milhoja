=============
User Manual
=============

++++++++++++++++++++++++++++
Operation Sepcification JSON
++++++++++++++++++++++++++++

Op Spec JSON documentation goes here.

+++++++++++++++++++++++++++
Task Function Specification
+++++++++++++++++++++++++++

The Task Function Specification is comprised of different sections that
allow the code generation tools to generate both the Task Function code
and the corresponding data item code. Each subsection of this section will detail
each section of the Task Function Specification and what needs to be inside each
one in order to be considered a 'valid' Task Function Specification.

This documentation is up to date with Milhoja JSON version 1.0.0 and milhoja pypackage
version 0.0.4. This documentation will always be updated to reflect primary standard
defined by Milhoja, which is the Milhoja JSON Format.

format
------

Contains Task Function Specification JSON metadata. Contains a list where
the first value in the list is the format type, and the second item is the
format version. Generally, this information is automatically inserted by the
code generation tools.

grid
----

Contains all grid specification information. Includes NXB, NYB, NZB, dimension,
and the number of guard cells.

task_function
-------------

This section contains all information specific to the task function.

language
^^^^^^^^

The source language of the task function. Available options include **C++** and
**Fortran**.

processor
^^^^^^^^^

Contains the hardware that the task function should run on. Available options
include **CPU** and **GPU**.

cpp_header
^^^^^^^^^^

The name of the header file for the task function.

cpp_source
^^^^^^^^^^

The name of the source file for a C++ task function. Also considered the C++ to
C layer when using the **Fortran** language.

c2f_source
^^^^^^^^^^

The name of the source file for the C to Fortran interoperability layer. Only
required if the `language <#language>`_ used is **Fortran**.

fortran_source
^^^^^^^^^^^^^^

The name of the fortran source file for the task function. Required if the
`language <#language>`_ being used is **Fortran**.

computation_offloading
^^^^^^^^^^^^^^^^^^^^^^

The computation offloading to use. Leave empty when `processor <#processor>`_ is **CPU**, 
and use **OpenACC** when **GPU** is the processor.

variable_index_base
^^^^^^^^^^^^^^^^^^^

The index base to use. Currently not implemented.

argument_list
^^^^^^^^^^^^^

The list of arguments for the task function. All arguments should be defined
inside of the `argument_specifications <#argument_specifications>`_ section.

argument_specifications
^^^^^^^^^^^^^^^^^^^^^^^

Contains the specifications for every argument found in `argument_list <#argument_list>`_.
Every argument specification will contain a "source" key that tells the code
generators what attributes to expect inside of the argument specification.

Source types
''''''''''''

Argument specifications include a wide array of different attributes depending
on what the "source" key contains. For each type of data source, the code generators
expect different attributes.

.. todo::
    * Explain different sources.
    * Write section that explains all valid types.

* external

A variable with an external source expects the following attributes: *type*,
*extents*. See `<#types>`_ for all possible values for *type*. Extents is an 
array of the format "(x, y, z, ...)" where the number of elements is the dimensionality
of the extents variable, and each value in the array is the size of the array.
Note that extents for variables is not yet implemented, however the generators
still expect that attribute to exist, so *extents* will always be "()".

* "tile" sources
    * tile_lo
    * tile_hi
    * tile_lbound
    * tile_ubound
    * tile_interior
    * tile_arrayBound
    * tile_deltas

There a large number of tile sources. Because "tile" sources are specific tile
keywords, all the information needed to properly generate code is built into
the code generation tools. Thus, "tile" sources don't require anything other
than the "source" keyword.

* lbound

The *lbound* source is used to store the lower bound of an array specified in the
Task Function Specification. It expects an *array* attribute, where the value
is the name of the variable inside of the specification associated with that
lbound.

* grid_data

The *grid_data* source is for grid variables. The *grid_data* expects the
attribute *structure_index*. It is a list, where the first value is "CENTER",
"FLUXX", "FLUXY", or "FLUXZ", and the second value is always 1. The *grid_data*
source also expects at least one of the following attributes: *variables_in*,
*variables_out*. Those attributes are a list containing a contiguous number index
range for an unk array.

* scratch

The *scratch* source is for variables that are intended to be used as scratch
arrays. Expects a *type*, *extents*, and *lbound* attribute.



subroutine_call_graph
^^^^^^^^^^^^^^^^^^^^^

Contains the call order for all of the functions specified in the `subroutines <#subroutines>`_
section. Consists of an ordered list of subroutine names. If multiple subroutines
can be called at once (i.e. using threads or streams), one can use a nested list 
of subroutine names instead of just one subroutine name.

data_item
---------

Contains supplemental information necessary for creating the data item for a
given task function. Includes:

    * **type** of data item, either "TileWrapper" or "DataPacket"
    * **byte_alignment** of data item variables. Only required for DataPackets.
    * **header** The name of the header file.
    * **source** The name of the source file.
    * **module** The name of the module file. Only required if `language <#language>`_ is **Fortran**.

subroutines
-----------

Cotnains key-value pairs, where the subroutine name is the key, and the value 
is a dictionary comprised of the information for generating calls to that subroutine
inside of the task function. The dictionary includes this information:

    * **interface_file** The interface or header file that the subroutine definition is in.
    * **argument_list** The ordered parameter list of the subroutine.
    * **argument_mapping** The mapping of subroutine arguments to `task function arguments <#argument_list>`_.

Array lower-bound arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Develop rules here.
