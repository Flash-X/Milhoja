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

* **byte-align**: The specified byte alignment of the memory system of the remote device that will use the data packet for offloaded computation. [16]
* **n-extra-streams**: The number of extra streams to use. Needed to allow for concurrent kernel execution. [0]
* **task-function-argument-list**: The order of arguments for the task function. This is required since JSONs are unordered.

constructor / thread-private-variables
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

tile-metadata
"""""""""""""
Tile-metadata is a dictionary containing metadata for each tile. The items in tile-metadata consist of a key, 
a specifier that may be referenced by items in the various array sections, and the value is an obtainable value 
from the Tile class. The list of obtainable values is a set of keywords contained in the data packet generator code.
The tile-metadata items are used for various items in the data array sections (`tile-in`, `tile-in-out`, `tile-out`, `tile-scratch`).
For example, lower and upper bounds for array indices are calculated using the information contained in this section.

Example:

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3_fx.json
    :linenos:
    :lines: 26-31

tile-in
"""""""
The data in this array is copied into the device being used. This dictionary consists of several keywords: 

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

Example:

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_1.json
    :linenos:
    :lines: 27-33

tile-in-out
"""""""""""
The data in this array is copied to the device being used, then data is copied back from the device to the same array. 
This section contains similar keywords to the other sections with some minor changes: 

* **type**: The data type of the items in the array to be copied in.
* **start-in**: The starting index of the array when being copied in.
* **start-out**: The starting index of the array when being copied out.
* **end-in**: The ending index of the array when being copied in.
* **end-out**: The ending index of the array when being copied out.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3.json
    :linenos:
    :lines: 31-40

tile-out
""""""""
This is the array to copy data back to. Again, this section contains similar keywords with minor changes:

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_1.json
    :linenos:
    :lines: 34-40

tile-scratch
""""""""""""
This section contains data arrays used as scratch space. Starts in the GPU and is not copied to the host or returned from the device.

* **type**: The data type of the items in the array to be copied in.
* **extents**: The extents of the array. This is an array of constants of size n. The extents in tile scratch includes the number of unknown variables at the end.

.. literalinclude:: ../../tools/cgkit_datapacket_generator/sample_jsons/DataPacket_Hydro_gpu_3.json
    :linenos:
    :lines: 5-22

JSON Abstraction Layer
----------------------

This is a python class responsible for abstracting the JSON file in such a way that the task function generator and DataPacket
generator can be given exactly what they need to generate their respective files without the need for having multiple or 
separate JSONs.
