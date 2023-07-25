Code Generation
===============

JSON
----

JSON Abstraction Layer
----------------------

This is a class responsible for abstracting the JSON file in such a way that the task function generator and data Packets
generator can be given exactly what they need to generate their respective files without the need for having multiple
or separate JSONs.

Task functions
--------------

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

To create the DataPacket subclass, the DataPacket generator makes use of CGKit, a code generation toolkit. The generator uses premade 
CGKit templates to assist with generating the final files for the packet, as well as using generated CGKit templates. To generate the 
various templates for assembling the final class, the DataPacket generator collects information for every item that needs to be in the 
data packet, as well as other information like the number of streams to use and the byte alignment, and formats it in a way that can 
be used by CGKit. This involves making two cgkit helper templates called **cg-tpl.datapacket_helpers.cpp** and 
**cg-tpl.datapacket_outer.cpp**. CGKit then uses those generated files along with the premade templates named **cg-tpl.datapacket.cpp** 
and **cg-tpl.datapacket_header.cpp** to generate the final output files for the new class, called **cgkit.datapacket.cpp** and 
**cgkit.datapacket.h**. 

If creating a packet for use with fortran, the DataPacket generator will create a few more files to allow the DataPacket items to 
be passed to a fortran task function. The DataPacket generator creates a C++ to C layer using CGKit. The generator will create new 
files using the same information to create the packets, called **cg-tpl.cpp2c_outer** and **cg-tpl.cpp2c_helper.cpp**, and combines 
it with either **cg-tpl.cpp2c_no_extra_queue.cpp** or **cg-tpl.cpp2c_extra_queue.cpp**. CGKit then takes these files, along with the 
premade C++ to C template called **cg-tpl.cpp2c.cpp**, and assembles it into the C++ to C layer found in **cgkit.cpp2c.cpp**. This is 
for another layer, the C to Fortran layer, since Fortran cannot interface with C++. The generation of this layer does not use CGKit, 
and **c2f.F90** is the file that is generated.

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