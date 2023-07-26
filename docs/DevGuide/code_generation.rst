Code Generation
===============

Not sure about the organization of this code generation doc.

JSON
----

| There are several sections that can be used inside of the JSON file. Most are optional, but some are not. 
| Default values for some items in each section will be specified in [brackets]. Otherwise, the item is required to be in the JSON file.

Packet Metadata
"""""""""""""""

| These options in the JSON file appear outside of any section and exist to help the generator with general packet information. 
| These are all possible parameters for the packet metadata:

* **byte-align**: The specified byte alignment. [16]
* **n-extra-streams**: The number of extra streams to use. [0]
* **task-function-argument-list**: The order of arguments for the task function. This is required since JSONs are unordered.

constructor / private-thread-variables
""""""""""""""""""""""""""""""""""""""
Non-tile-specific data goes here. Note that the number of tiles is automatically inserted into the data packet, 
so those do not need to be specified in the JSON file. Any items in general must be passed into the data packet 
constructor, and the data packet will take ownership of the items and copy them into its own memory to manage it. 
We are assuming the items will remain valid throughout the life cycle of the entire packet. This section is 
used for non-tile-specific variables that are used in the task functions.

.. code-block::

    "constructor": {
        "dt": "real"
    }

tile-metadata
"""""""""""""
Tile-metadata is a dictionary containing metadata for each tile. The items in tile-metadata consist of a key, 
a specifier that may be referenced by items in the various array sections, and the value is an obtainable value 
from the Tile class. The list of obtainable values is a set of keywords contained in the data packet generator code.
The tile-metadata items are used for various items in the data array sections (`tile-in`, `tile-in-out`, `tile-out`, `tile-scratch`).
For example, lower and upper bounds for array indices are calculated using the information contained in this section.

Example:

.. code-block::

    "tile-metadata": {
        "tile_deltas": "deltas",
        "tile_lo": "lo",
        "tile_hi": "hi"
    }

tile-in
"""""""
The data in this array is copied into the device being used. This dictionary consists of several keywords: 

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

Example:

.. code-block::

    "tile-in": {
        "type": "real",
        "start": 0,
        "end": 8,
        "extents": ["8 + 2 * 1", "8 + 2 * 1", "1 + 2 * 0"]
    }

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

.. code-block::

    "tile-in-out": {
        "type": "real",
        "start-in": 0,
        "start-out": 8,
        "end-in": 0,
        "end-out": 7,
        "extents": ["8 + 2 * 1", "8 + 2 * 1", "1 + 2 * 0"]
    }

tile-out
""""""""
This is the array to copy data back to. Again, this section contains similar keywords with minor changes:

* **type**: The data type of the items in the array to be copied in.
* **start**: The starting index of the array.
* **end**: The ending index of the array.
* **extents**: The extents of the array. This is an array of constants of size n. This determines the dimensions of the data array.

.. code-block::

    "tile-out": {
        "type": "real",
        "start": 0,
        "end": 7,
        "extents": ["8 + 2 * 1", "8 + 2 * 1", "1 + 2 * 0"]
    }

tile-scratch
""""""""""""
This section contains data arrays used as scratch space. Starts in the GPU and is not copied to the host or returned from the device.

* **type**: The data type of the items in the array to be copied in.
* **extents**: The extents of the array. This is an array of constants of size n. The extents in tile scratch includes the number of unknown variables at the end.

.. code-block::

    "tile-scratch": {
        "type": "real",
        "extents": ["8 + 2 * 1", "8 + 2 * 1", "1 + 2 * 0", "5"]
    }


JSON Abstraction Layer
----------------------

This is a python class responsible for abstracting the JSON file in such a way that the task function generator and DataPacket
generator can be given exactly what they need to generate their respective files without the need for having multiple or 
separate JSONs.

Task functions
--------------

Task function generation doc goes here?

Data Packets
------------

If the Flash-X recipe determines that certain tasks need to be executed on an external device, the Milhoja Runtime will eventually call 
the DataPacket generator for creating a derived class of DataPacket in order to send information to an external device. When the
generator is called, it will create various files based on the information passed to it. In order to generate a new DataPacket 
subclass, the DataPacket generator will need:

* The byte sizes for each data type used by an item that needs to be put in a data packet.
* The language that is being used to run the task function. Either 'cpp' or 'fortran'.
* Various bits of information for each item in the data packet (see :doc:`JSON`).

Using that information, the DataPacket generator will create a new subclass for passing information to an external device.

Packet Generation Steps
"""""""""""""""""""""""

1. The DataPacket generator takes in a language specifier, a byte sizes JSON, and DataPacket :doc:`JSON``. 

2. The DataPacket generator loads each JSON and creates any extra information it needs for DataPacket class generation.

3. The generator moves through each section in the :doc:`JSON`` in a specified order [`constructor`, `tile-metadata`, `tile-in`, 
`tile-in-out`, `tile-out`, and `tile-scratch`], sorts the section by size, then generates formatted links, connectors, 
and param strings and stores them in dictionaries for later use. The links, connectors, and params from the DataPacket 
generator act as the implementation for the DataPacket. Since the JSON format does not have an inherent order, 
the DataPacket generator uses the specified order to group related variables together in the new DataPacket subclass. 
However, even if the variables are grouped together, this does not mean that they will appear in the packet in the same 
order. That depends on the sizes of each item in the packet.

4. Once the generator has gone through each section in the JSON, it will write the strings in the dictionaries to cgkit 
template files for use by cgkit. These files are **cg-tpl.datapacket_helpers.cpp** and **cg-tpl.datapacket_outer.cpp**.

5. Once the files have been generated, the DataPacket generator calls CGKit to insert each param, connector, and link into 
the premade template files named **cg-tpl.datapacket.cpp** and **cg-tpl.datapacket_header.cpp**, resulting in the completed
implementation of the new DataPacket class. The output files are named **cgkit.datapacket.cpp** and **cgkit.datapacket.h**. 

6. If the 'fortran' language is specified, the DataPacket generator will call two more functions to create interoperability 
layers for the new DataPacket. One is the C++ to C layer, and the other is the C to Fortran layer. The C to Fortran layer and 
the C++ to C layer are created using the same inputs used for the DataPacket.

7. The C to Fortran layer generation creates a new Fortran 90 file that converts the C pointers and variable members in the 
DataPacket to Fortran based variables. This file is named **c2f.f90**.

8. The C++ to C layer is created using CGKit. Two more template files are generated and are combined with pre-existing template 
files to create the layer. The generated template files are named **cg-tpl.cpp2c_outer** and **cg-tpl.cpp2c_helper.cpp** and 
the existing templates are **cg-tpl.cpp2c_no_extra_queue.cpp** or **cg-tpl.cpp2c_extra_queue.cpp** and **cg-tpl.cpp2c.cpp**. 

Using the information from the JSON input, the DataPacket generator will implement methods from the base DataPacket class as necessary.
The methods `extraAsynchronousQueue()` and `releaseExtraQueue()` are given derived class implementations if there are more than 
0 n-extra-streams. The clone method is overridden, using the arguments from the `constructor` section to create a new packet, 
satisfying the prototype design pattern. The `unpack()` function uses information from `tile-in`, `tile-in-out`, and `tile-out` to 
unpack the information from device memory back into host memory.

The `pack()` function is generated using three distinct phases. The first is the size determination phase, where the size of each item 
in the packet is determined, as well as the overall size of the packet. The next phase is the pointer determination phase. Since the 
packet uses a SoA pattern, the pointer determination phase gets the start of each array of pointers for each item in the packet. The 
size of each item's pointer array is equal to the number of tiles. The last phase is the memcpy phase, where pointers from the host 
memory are copied into pinned memory.

Data Mapping
------------
Every item specified in a JSON file will have a mapping in the data packet associated with it. The data packet generator will create
multiple variables for use within and outside of the data packet. The variables shall be called the name of the associated item followed by 
a prefix and a suffix.

For items in host memory, each item in the JSON will have an associated variable in the data packet that starts with the prefix '_',
followed by the name of the item, followed by the suffix '_h'. Items contained in the 'constructor'/'private-thread-variables' are the 
only variables contained in the data packet that have associated host variables in the data packet. Example: 'dt' -> '_dt_h'.

For items in device memory, each item in the JSON will have an associated variable in the data packet that starts with the prefix '_',
followed by the name of the item, followed by the suffix '_d'. Every item in the JSON will have an associated device pointer.
Example: 'dt' -> '_dt_d'. 

Items in the tile-in and tile-in-out sections have pinned memory pointers associated with them in the data packet. This starts with the 
prefix '_', followed by the name of the item, followed by the suffix '_p'. Example: 'Uin' -> '_Uin_p'.

When creating a packet using the 'cpp' language option, each item in tile-in, tile-in-out, tile-out, and tile-scratch will have FArrayND 
device memory pointers associated with them. The name of the pointer starts with the prefix '_f4_', followed by the name of the item,
followed by the suffix '_d'.