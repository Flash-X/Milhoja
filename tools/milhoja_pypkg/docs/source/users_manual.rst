=============
User Manual
=============

+++++++++++++++++++++++
Operation Specification
+++++++++++++++++++++++

.. todo::
    * Write Op Spec Specification here.

+++++++++++++++++++++++++++
Task Function Specification
+++++++++++++++++++++++++++

The Task Function Specification is comprised of different sections that allow
the code generation tools to generate both the Task Function code and the corresponding
data item code. Each subsection of this section will detail each part of the Task
Function Specification and what needs to be inside each one to be considered
a 'valid' Task Function Specification.

This documentation is up to date with Milhoja JSON version 1.0.0 and milhoja pypackage
version 0.0.4. This documentation will always be updated to reflect primary standard
defined by and used Milhoja, which is the Milhoja JSON Format.

While it is possible to load a raw Task Function Specification and use it, it is
highly recommended that you load the JSON using 

Task Function Specification Requirements
----------------------------------------

.. todo::
    * Write requirements...

format
------

Contains Task Function Specification JSON metadata. Contains a list where
the first value in the list is the format type, and the second item is the
format version. Generally, this information is automatically inserted by the
code generation tools.

Example:

.. code-block:: json

    {
        "format": ["Milhoja-JSON", "1.0.0"]
    }

grid
----

Contains all grid specification information. Includes NXB, NYB, NZB, dimension,
and the number of guard cells.

Example:

.. code-block:: json

    {
        "grid": {
            "dimension": 2,
            "nguardcells": 6,
            "nxb": 16,
            "nyb": 16,
            "nzb": 1
        },
    }

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
required if the :ref:`users_manual:language` used is **Fortran**.

fortran_source
^^^^^^^^^^^^^^

The name of the fortran source file for the task function. Required if the
:ref:`users_manual:language` being used is **Fortran**.

computation_offloading
^^^^^^^^^^^^^^^^^^^^^^

The computation offloading to use. Leave empty when :ref:`users_manual:processor` is **CPU**, 
and use **OpenACC** when **GPU** is the processor.

variable_index_base
^^^^^^^^^^^^^^^^^^^

The index base to use. Currently not implemented.

argument_list
^^^^^^^^^^^^^

The list of arguments for the task function. All arguments should be defined
inside of the :ref:`users_manual:argument_specifications` section.

argument_specifications
^^^^^^^^^^^^^^^^^^^^^^^

Contains the specifications for every argument found in :ref:`users_manual:argument_list`.
Every argument specification will contain a "source" key that tells the code
generators what attributes to expect inside of the argument specification.

Source types
''''''''''''

Argument specifications include a wide array of different attributes depending
on what the "source" key contains. For each type of data source, the code generators
expect different attributes.

external
********

A variable with an external source expects the following attributes: *type*,
*extents*. See :ref:`users_manual:types` for all possible values for *type*. Extents is an 
array of the format "(x, y, z, ...)" where the number of elements is the dimensionality
of the extents variable, and each value in the array is the extents of that array.
Note that extents for variables is not yet implemented, however the generators
still expect that attribute to exist, so *extents* will always be "()".

Any external variables are passed in by value to the Data Item constructor,
meaning that a deep copy of the variable is made and stored within the Data
Item. Therefore, external variables cannot be changed once they are passed into
the Data Item, and thus thread-private variables.

Example:

.. code-block:: json

    {
        "argument_specifications": {
            "external_Hydro_dt": {
                "type": "real",
                "extents": "()",
                "source": "external"
            },
        }
    }

tile_metadata sources
*********************

All tile_metadata sources:

    * tile_lo
    * tile_hi
    * tile_lbound
    * tile_ubound
    * tile_interior
    * tile_arrayBound
    * tile_deltas

There a large number of tile_metadata sources. Because these sources are specific tile
keywords, all the information needed to properly generate code is built into
the code generation tools. Thus, tile_metadata sources don't require anything other
than the "source" keyword.

Examples:

.. code-block:: json

    {
        "argument_specifications": {
            "tile_arrayBounds": {
                "source": "tile_arrayBounds"
            },
            "tile_deltas": {
                "source": "tile_deltas"
            },
            "tile_interior": {
                "source": "tile_interior"
            },
            "tile_lbound": {
                "source": "tile_lbound"
            },
            "tile_lo": {
                "source": "tile_lo"
            }
        }
    }

lbound
******

The *lbound* source is used to store the lower bound of an array specified in the
Task Function Specification. It expects an *array* attribute, where the value
is the name of the variable inside of the specification associated with that
lbound.

Example: [#]_

.. code-block:: json

    {
        "argument_specifications": {
            "scratch_hydro_op1_auxC": {
                "source": "scratch",
                "type": "real",
                "extents": "(18, 18, 18)",
                "lbound": "(tile_lo) - (1, 1, 1)"
            }
            "lbdd_scratch_hydro_op1_auxC": {
                "source":      "lbound",
                "array":       "scratch_hydro_op1_auxC"
            }
        }
    }

.. [#] Notice that "scratch_hydro_op1_auxC" must be defined for the lbound.

grid_data
*********

The *grid_data* source is for grid variables. The *grid_data* expects the
attribute *structure_index*. It is a list, where the first value is "CENTER",
"FLUXX", "FLUXY", or "FLUXZ", and the second value is always 1. The *grid_data*
source also expects at least one of the following attributes: *variables_in*,
*variables_out*. Those attributes are a list containing a contiguous number index
range for an unk array.

Example:

.. code-block:: json

    {
       "argument_specifications": {
            "CC_1": {
                "source": "grid_data",
                "structure_index": ["CENTER", 1],
                "variables_in": [1, 18],
                "variables_out": [1, 18]
            },
            "FLX_1": {
                "source": "grid_data",
                "structure_index": ["FLUXX", 1],
                "variables_in": [1, 5],
                "variables_out": [1, 5]
            }
        }
    }

scratch
*******

The *scratch* source is for variables that are intended to be used as scratch
arrays. Expects a *type*, *extents*, and *lbound* attribute.

.. code-block:: json

    {
        "argument_specifications": {
            "scratch_Hydro_hy_uPlus": {
                "source": "scratch",
                "type": "real",
                "extents": "(28,28,1,7)",
                "lbound": "(tile_lbound, 1)"
            },
            "scratch_Hydro_xCenter_fake": {
                "source": "scratch",
                "type": "real",
                "extents": "(1)",
                "lbound": "(1)"
            }
        }
    }

subroutine_call_graph
^^^^^^^^^^^^^^^^^^^^^

Contains the call order for all of the functions specified in the :ref:`users_manual:subroutines`
section. Consists of an ordered list of subroutine names. If multiple subroutines
can be called at once (i.e. using threads or streams), one can use a nested list 
of subroutine names instead of just one subroutine name.

Example:

.. code-block:: json

    {
        "subroutine_call_graph": [
            "Hydro_prepBlock",
            "Hydro_advance"
        ]
    }

data_item
---------

Contains supplemental information necessary for creating the data item for a
given task function. Includes:

    * **type** of data item, either "TileWrapper" or "DataPacket"
    * **byte_alignment** of data item variables. Only required for DataPackets.
    * **header** The name of the header file.
    * **source** The name of the source file.
    * **module** The name of the module file. Only required if :ref:`users_manual:language` is **Fortran**.

Example:

.. code-block:: json

    {
        "data_item": {
            "type": "TileWrapper",
            "byte_alignment": 16,
            "header": "TileWrapper_cpu_taskfn_0.h",
            "module": "TileWrapper_cpu_taskfn_0_mod.F90",
            "source": "TileWrapper_cpu_taskfn_0.cxx"
        },
    }

subroutines
-----------

Cotnains key-value pairs, where the subroutine name is the key, and the value 
is a dictionary comprised of the information for generating calls to that subroutine
inside of the task function. The dictionary includes this information:

    * **interface_file** The interface or header file that the subroutine definition is in.
    * **argument_list** The ordered parameter list of the subroutine.
    * **argument_mapping** The mapping of subroutine arguments to :ref:`users_manual:argument_specifications`


types
-----

The list of valid types includes:

* **bool**, **logical**
* **int**, **integer**, **unsigned int**
* **real**, **milhoja::Real**

Array lower-bound arguments
---------------------------

The current set of rules for writing lbound attributes is as follows:

1. Lbound attribute values will be comprised of a parenthesis composed string
   with comma separated values.

2. Lbounds are allowed to contain **tile_lo**, **tile_hi**, **tile_lbound**,
   and **tile_ubound** keywords. These are considered arrays of size 3, so they
   can only be used in variables of array size 3 or higher.

3. Lbound strings are allowed to use non-nested mathematical expressions between
   parenthesis. However, the operands must be the same size.

4. Negative values are allowed for lbound strings.

Examples of valid formats include:
    * (1, 2, -3, 4)
    * (tile_lo, 1)
    * (1, tile_lbound),
    * (tile_lo) - (1, 1, 1)
    * (tile_lbound, 1) + (1, 3, 4, 5)
    * (tile_lo, tile_lo) - (tile_lbound, tile_lbound)
    * (1, 2, 3) + (4, 5, 6) - (2, 2, 2) * (1, 2, 3)

Examples of invalid lbound formats:
    * (2, (3-4-6), 2, (92))
    * (tile_lo, tile_lo) - (tile_lo)
    * (1, 2, 3) + (tile_lo, 2, 3, 4)

