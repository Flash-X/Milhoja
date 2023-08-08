Milhoja Input Interface
=======================

This is a living document detailing each JSON input that the Milhoja Runtime uses.

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
