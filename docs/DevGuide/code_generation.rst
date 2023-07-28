Code Generation
===============

This document is a living document going in depth into the various code generation systems used in the 
Milhoja Runtime. Due to how code generation is interspersed throughout the MIlhoja Runtime, this document 
is organized by the various functionalities that code generation serves.

Task functions
--------------

Task function generation doc goes here?

Data Packets
------------

If the Flash-X recipe determines that certain tasks need to be executed on a remote device, the Milhoja Runtime will eventually call 
the DataPacket generator for creating a derived class of DataPacket in order to send information to a remote device. When the
generator is called, it will create new files based on the information passed to it. To do this, the DataPacket generator will need:

* The byte sizes for every data type used in the DataPacket JSON input.
* The language that is being used to run the task function. Either 'cpp' or 'fortran'.
* A JSON input containing all information needed for each item in a data packet. 
  For specifications on this, read the DataPacket JSON section in the UserManual.

Using that information, the DataPacket generator will create a new subclass for passing information to a remote device.

Packet Generation Steps
"""""""""""""""""""""""

1. The DataPacket generator takes in a language specifier, a byte sizes JSON, and DataPacket. 

2. The DataPacket generator loads each JSON and creates any extra information it needs for DataPacket class generation.

3. The generator moves through each section in the DataPacket JSON in a specified order [`constructor`, `tile-metadata`, `tile-in`, 
   `tile-in-out`, `tile-out`, and `tile-scratch`], sorts the section by size, then generates formatted links, connectors, 
   and param strings and stores them in dictionaries for later use. The links, connectors, and params from the DataPacket 
   generator act as the implementation for the DataPacket. Since the JSON format does not have an inherent order, 
   the DataPacket generator uses the specified order to group related variables together in the new DataPacket subclass. 
   However, even if the variables are grouped together, this does not mean that they will appear in the packet in the same 
   order. That depends on the sizes of each item in the packet.

   a. Using the information from the JSON input, the DataPacket generator will implement methods from the base DataPacket class as necessary.
      The methods `extraAsynchronousQueue()` and `releaseExtraQueue()` are given derived class implementations if there are more than 
      0 n-extra-streams. The clone method is overridden, using the arguments from the `constructor` section to create a new packet, 
      satisfying the prototype design pattern. The `unpack()` function uses information from `tile-in`, `tile-in-out`, and `tile-out` to 
      unpack the information from device memory back into host memory.

   b. The `pack()` function is generated using three distinct phases. The first is the size determination phase, where the size of each item 
      in the packet is determined, as well as the overall size of the packet. The next phase is the pointer determination phase. Since the 
      packet uses a SoA pattern, the pointer determination phase gets the start of each array of pointers for each item in the packet. The 
      size of each item's pointer array is equal to the number of tiles. The last phase is the memcpy phase, where pointers from the host 
      memory are copied into pinned memory.

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

Data Mapping
------------
Every item specified in a JSON file will have a mapping in the data packet associated with it. The data packet generator will create
multiple variables for use within and outside of the data packet. The variables shall be called the name of the associated item followed by 
a prefix and a suffix.

For items in host memory, each item in the JSON will have an associated variable in the data packet that starts with the prefix '_',
followed by the name of the item, followed by the suffix '_h'. Items contained in the 'constructor'/'thread-private-variables' are the 
only variables contained in the data packet that have associated host variables in the data packet. Example: 'dt' -> '_dt_h'.

For items in device memory, each item in the JSON will have an associated variable in the data packet that starts with the prefix '_',
followed by the name of the item, followed by the suffix '_d'. Every item in the JSON will have an associated device pointer.
Example: 'dt' -> '_dt_d'. 

Items in the tile-in and tile-in-out sections have pinned memory pointers associated with them in the data packet. This starts with the 
prefix '_', followed by the name of the item, followed by the suffix '_p'. Example: 'Uin' -> '_Uin_p'.

When creating a packet using the 'cpp' language option, each item in tile-in, tile-in-out, tile-out, and tile-scratch will have FArrayND 
device memory pointers associated with them. The name of the pointer starts with the prefix '_f4_', followed by the name of the item,
followed by the suffix '_d'.

    * NOTE: When should we be using FArray3D / FArray2D / FArray1D instead of FArray4D? Is this based on dimensionality of the problem?

