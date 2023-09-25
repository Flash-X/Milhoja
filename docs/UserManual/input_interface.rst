Milhoja Input Interface
=======================

This is a living document detailing each JSON input that the Milhoja Runtime uses. This is the default input method 
that is specified by Milhoja as an initial option to pass data to its code generators. However, in the future there 
may be a more concrete method of specifying and parsing data that is defined by the application to suit their purposes.
In this case, the code generators have a predefined set of information it needs in order to properly generate a Task 
Function or DataPacket that the application would need to satisfy when creating its own method of outlining and 
parsing information, so that the generators have all of the necessary information to generate code.

DataPacket JSON
---------------

The DataPacket JSON is comprised of various fields that determine how a new DataPacket subclass is generated.

Packet Metadata
"""""""""""""""

| These options in the JSON file appear outside of any section and exist to help the generator with general packet information. 
| These are all possible parameters for the packet metadata:

* **byte_align**: The specified byte alignment of the memory system of the remote device that will use the data packet for offloaded computation. [16]
* **n_extra_streams**: The number of extra streams to use. Needed to allow for concurrent kernel execution. [0]
* **task_function_argument_list**: The order of arguments for the task function. This is required since JSONs are unordered.

constructor / thread_private_variables
""""""""""""""""""""""""""""""""""""""
Non-tile-specific data goes here. Note that the number of tiles is automatically inserted into the data packet, 
so those do not need to be specified in the JSON file. Any variables in general must be passed into the data packet 
constructor where the data packet will copy them into its own memory to manage it. This means that the data packet "owns"
the variable and will be available for use throughout the lifetime of the data packet. We are assuming the variables 
will remain valid throughout the life cycle of the entire packet. This section is used for non-tile-specific variables 
that are used in the task functions. The format of an item included in this section is `name: data_type`.

TODO: The current iteration of the data packet assumes that any variables in the constructor / thread-private-variables 
section cannot be changed after the packet is instantiated. Is there a scenario where we might want a task function that 
can change these values? The variables might be considered datapacket-private then if there is no such scernario.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3_fx.json
    :linenos:
    :lines: 23-25

tile_metadata
"""""""""""""
Tile-metadata is a dictionary containing metadata for each tile. The items in tile-metadata consist of a key, 
a specifier that may be referenced by items in the various array sections, and the value is an obtainable value 
from the Tile class. The list of obtainable values is a set of keywords contained in the data packet generator code.
The tile-metadata items are used for various items in the data array sections (`tile-in`, `tile-in-out`, `tile-out`, `tile-scratch`).
For example, lower and upper bounds for array indices are calculated using the information contained in this section.

TODO: The existing tile_metadata section does not have a concrete way to use milhoja Grid interface functions. 
There needs to be some kind of mapping between keywords and Grid interface functions. The tile_metadata section 
would need these keywords to be contained within the `source` value, and any other potential keywords necessary 
for calling a grid interface function (the axis, bounds, or edge, for example). 

My initial idea is to have a small set of keywords that point to grid functions, and to have extra information 
contained within tile_metadata keys that can pass in parameters to those specific Grid functions. For example, 
a tile_metadata variable can contain the name of the corresponding application variable, pointing to a dictionary 
containing a `source` attribute (like what Jared has done) that points to a specific function call based on a map 
contained within the code generator. Information pertaining to that function call would be contained within that
dictionary as necessary (e.g. Axis, bounds, edge). This seems like the most effective way to me at the moment to 
be able to easily add new Grid function calls to the data packet generator without substantial changes to the 
actual code.

Some things to keep in mind: 1) While we chould have a keyword for every possible Grid function and parameter 
combination, this becomes unwieldy and unmaintainable as we add more possible keyword -> function maps or parameters. 
So we want to avoid that. 2) This is more generator related, but if we go down the route of having specific information 
included in the tile_metadata, then the mapping needs to be formatted in such a way that the strings of functions use 
every possible parameter inside of the string. 3) How will this work with the c2f interface? 4) It's very possible that 
some grid functions are not as simple as just inserting the function call into the source code. Should all of the necessary 
code to use that function be included inside of that mapping? 5) When calling Grid functions, what information is directly 
tied to the tile iterator, and what information should be specified by the application / user?

All possible tile_metadata keys:

* **tile_lo**: The index of the lower corner of the interior of the associated region
* **tile_hi**: The index of the upper corner of the interior of the associated region
* **tile_level**: The refinement level in which the associated region is defined
* **tile_deltas**: The array of the mesh spacing along all directions
* **tile_lbound**: the index associated with the lower corner of the data array associated with the region
* **tile_ubound**: the index associated with the upper corner of the data array associated with the region

Example:

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3_fx.json
    :linenos:
    :lines: 26-31

tile_in
"""""""
TODO: The current JSON setup for tile_in / tile_in_out / tile_out / tile_scratch obtains the size of the 
array by subtracting the start and ends of the array and adding 1. However, if we want the starting offset 
to be anything other than 0, this will not work. For example, in the Runtime tests, there exists a packet that
needs to be size 1, and the start and end indices need to be 1. Because the starting and ending index are 1,
but the size of the array is also 1, the data packet tries to use an offset outside of the array's memory,
causing memory access violations. There are 2 proposed fixes for this problem. 

One is to allow additional functionality to the FArray4D class to be able to offset its starting index. 
Unfortunately, I don't believe this solution solves the issue for Fortran packets, since a Fortran packet does
not use FArray4D objects to store arrays. There would need to be adjustments made to the DataPacket generator 
for this to work properly.

The second solution is to split the NUNKS and variable masking offsets of the data packet. This allows for 
any desired variable masking while also being able to set a specific size for the array. This solution is 
essentially what the hand-written data packets used, where the size of the array was determined by NUNKVAR
and the variable masking was set using a separate variable masking function. Only this time, only the data 
packet itself would have access to the variable masking information, keeping it self contained. The problem 
with this solution is that it's much more error prone when creating DataPacket jsons. The upside is that it's
an easy solution to this problem.

The data in this array is copied into the device being used. This dictionary consists of several keywords: 

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of integer constants of size n. This determines the dimensions of the data array.

Example:

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_1.json
    :linenos:
    :lines: 27-33

tile_in_out
"""""""""""
The data in this array is copied to the device being used, then data is copied back from the device to the same array. 
Arrays in tile_in_out must have specified start_in, end_in, start_out, and end_out indices, since the array may possibly 
return less (or more?) data than it was packed with. Because of this, arrays specified in here do not currently support 
different spatial dimensions for packing and unpacking, and only support changing the number of variables.

* **type**: The data type of the items in the array to be copied in.
* **start_in**: The starting index of the array when being copied in.
* **start_out**: The starting index of the array when being copied out.
* **end_in**: The ending index of the array when being copied in.
* **end_out**: The ending index of the array when being copied out.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3.json
    :linenos:
    :lines: 31-40

tile_out
""""""""
This is the array to copy data back to. Again, this section contains similar keywords with minor changes:

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of integer constants of size n. This determines the dimensions of the data array.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_1.json
    :linenos:
    :lines: 34-40

tile_scratch
""""""""""""
This section contains data arrays used as scratch space with one scratch array provided for each tile in the data packet. 
Starts in the GPU and is not copied to the host or returned from the device.

* **type**: The data type of the items in the array to be copied in.
* **extents**: The extents of the array. This is an array of integer constants of size n. The extents in tile scratch 
               includes the number of unknown variables at the end.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3.json
    :linenos:
    :lines: 5-22

JSON Abstraction Layer
----------------------

This is a python class responsible for abstracting the JSON file in such a way that the task function generator and DataPacket
generator can be given exactly what they need to generate their respective files without the need for having multiple or 
separate JSONs.

Given an input data file and a specific file format reader, the JSON abstraction layer should be able to provide the 
necessary information needed to generate a packet or task function without the code generators knowing how the inforamtion 
was obtained. 

I'm going to list some requirements here so I don't forget.

1. Any given data section containing variables for a generated DataPacket can have 0 to N total variables.
2. 
