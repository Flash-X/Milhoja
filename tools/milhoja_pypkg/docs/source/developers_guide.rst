=================
Developers' Guide
=================

Developer Environment
---------------------

Development with tox
^^^^^^^^^^^^^^^^^^^^
The coverage reports for this package are managed with `tox
<https://tox.wiki/en/latest/index.html>`_, which can be used for CI work among
other work.  However, the same ``tox`` setups can be used by developers if so
desired.  This can be useful since ``tox`` will automatically setup and manage
dedicated virtual environments for the developer.  The following guide can be
used to setup ``tox`` as a command line tool on an individual platform in a
dedicated, minimal virtual environment and is based on the a `webinar
<https://www.youtube.com/watch?v=PrAyvH-tm8E>`_ by Oliver Bestwalter.  I
appreciate his solution as there is no need to activate the virtual environment
in order to use ``tox``.

Developers that would like to use ``tox`` should learn about the tool so that,
at the very least, they understand the difference between running ``tox`` and
``tox -r``.

To create a python virtual environment based on a desired python dedicated to
hosting ``tox``, execute some variation of

.. code-block:: console

    $ cd
    $ deactivate (to deactivate the current virtual environment if you are in one)
    $ /path/to/desired/python --version
    $ /path/to/desired/python -m venv $HOME/.toxbase
    $ ./.toxbase/bin/pip list
    $ ./.toxbase/bin/python -m pip install --upgrade pip
    $ ./.toxbase/bin/pip install --upgrade setuptools
    $ ./.toxbase/bin/pip install tox
    $ ./.toxbase/bin/tox --version
    $ ./.toxbase/bin/pip list

To avoid the need to activate ``.toxbase``, we setup ``tox`` in ``PATH`` for
use across all development environments that we might have on our system. In
the following, please replace ``.bash_profile`` with the appropriate shell
configuration file and tailor to your needs.

.. code-block:: console

    $ mkdir $HOME/local/bin
    $ ln -s $HOME/.toxbase/bin/tox $HOME/local/bin/tox
    $ vi $HOME/.bash_profile (add $HOME/local/bin to PATH)
    $ . $HOME/.bash_profile
    $ which tox
    $ tox --version

If the environment variable ``COVERAGE_FILE`` is set, then this is the coverage
file that will be used with all associated work.  If it is not specified, then
the coverage file is ``.coverage_milhoja``.

No work will be carried out by default with the calls ``tox`` and ``tox -r``.

The following commands can be run from the directory that contains this
package's ``tox`` configuration file

.. code-block:: console

    /path/to/milhoja_pypkg/tox.ini

* ``tox -r -e coverage``

  * Execute the full test suite for the package and save coverage results to
    the coverage file
  * The test runs the package code in the local clone rather than code
    installed into python so that coverage results accessed through web
    services such as Coveralls are clean and straightforward
* ``tox -r -e nocoverage``

  * Execute the full test suite for the package using the code installed into
    python
* ``tox -r -e check``

  * This is likely only useful for developers working on a local clone
  * Run several checks on the code to report possible issues
  * No files are altered automatically by this task
* ``tox -r -e html``

  * Generate and render the package's documentation locally in HTML
* ``tox -r -e pdf``

  * Generate and render the package's documentation locally as a PDF file

Additionally, you can run any combination of the above such as ``tox -r -e
report,coverage``.

Manual Developer Testing
^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to test manually outside of ``tox``, which could be useful for
testing at the level of a single test.

The following example shows how to run only a single test case using the
``coverage`` virtual environment setup by ``tox``.

.. code-block:: console

    $ cd /path/to/milhoja_pypkg
    $ tox -r -e coverage
    $ . ./.tox/coverage/bin/activate
    $ which python
    $ python --version
    $ pip list
    $ python -m unittest milhoja.TestTaskFunction

Code Generation
---------------

This document is a living document detailing the code generation systems that come bundled
with the Milhoja pypackage.

There are two major types of codes that the Milhoja pypackage can generate for the
user when provided with a :ref:`users_manual:Task Function Specification`. These are
:ref:`developers_guide:Task Functions` and :ref:`developers_guide:Data Items`.
These two types of codes are considered a pair, so one JSON input should be used
to generate both at the same time. One is not guaranteed to work without the other.
Ultimately, the application using this package decides what code needs to be generated.

Task Functions
^^^^^^^^^^^^^^

Task Functions are responsible for using :ref:`developers_guide:Data Items` to
call the subroutines specified inside of the :ref:`users_manual:Task Function Specification`,
in the order that they are contained in the call graph.

Because it is necessary for the Task Function to call all subroutines contained
inside of a call graph, it needs some sort of an argument mapping for each specified
subroutine. This is necessary so that it can setup data pointers contained inside
of the associated Data Item to pass into each subroutine call at the appropriate spot.
Without the data mapping contained within the Task Function Specification, the
Task Function Generator would have no idea how to pass arguments into each subroutine,
outside of using string matching. However, relying on strings to pass in arguments
is prone to errors and difficult to debug, so we avoid that approach in all code
generators.

Generation
''''''''''

1. The milhoja library function generate_task_function is called, and a Task Function
   Specification, directory destination, overwrite flag, indent size, and logger
   are passed into it. The generate_task_function routine will choose the appropriate
   internal Task Function Generator class based on the details outline in the
   Task Function Specification.

    a. Depending on the Task Function Specification details, the "C++ to C" and
    "C to Fortran" layers may also be generated after the Task Function is generated.

2. Once a Task Function Generator class is chosen, a new instance of it will be
   created and the generate_header_code & generate_source_code member functions
   will be called in order to generate code.

3. Each Task Function Generator class will move through each argument defined inside
   of the Task Function Specification, inserting it into the argument list and
   setting it up inside of the Task Function. 

4. Once the generator has set up each of the arguments, it will then move through
   the subroutine graph in order to generate the call order for the graph.

5. If the specified language inside of the Task Function Specification is fortran,
   the Task Function Generator will also generate a module that binds the C++ to C
   layer function as a fortran method.

Data Items
^^^^^^^^^^

Generated Data Items are responsible for holding the information needed by the
Task Function, and work in tandem with the Milhoja runtime if using any device
offloading. There are two types of Data Items, :ref:`developers_guide:Tile Wrappers`
and :ref:`developers_guide:Data Packets`.

Tile Wrappers
"""""""""""""

Tile Wrappers are data items that contain a tile reference as well as thread-private
variables. Generally, Tile Wrappers are used for Task Functions that do not
require device offloading.

TileWrapper Requirements
''''''''''''''''''''''''

1. The base TileWrapper class shall be derived from DataItem and written such that an
   instantiation of it can be used directly as the prototype for a TF that has no 
   thread-private variables. The class interface must contain a means for TFs to
   access tile metadata via objects instantiated from TileWrapper or from objects
   instantiated from classes derived from TileWrapper.

2. For those tile-based TFs that use "external" variables, a derived custom version
   of TileWrapper shall be written by a Milhoja code generator in a way analogous
   to the generation of custom DataPacket code. The interface shall allow for access
   of tile metadata and external variables by the TF and such that the TF code
   generator can write the code needed to access all such variables.

3. If thread-private scratch data is needed by a tile-based TF, then the derived
   custom version of TileWrapper shall be written to allocate a static pool of
   scratch memory of sufficient size structured such that each activated thread
   in the team to which the TF is assigned can gain exclusive access to its scratch
   variable using its unique thread index.  The derived class's interface shall
   include static routines for acquiring and releasing the scratch memory.
   This interface shall be such that it allows calling code to determine when to
   acquire and release the scratch rather than impose, for instance, that scratch
   memory management must occur as part of or just before/after a Milhoja invocation.

4. The interface of the TileWrapper class shall be made and maintained as similar
   as possible to that of the DataPacket class so that the use of prototypes and
   cloning by the runtime and calling code are as similar as possible. This should
   ease learning and using the design as well as improve maintainability of codes
   that use Milhoja.

.. todo::
  * Write tile wrapper overview.
  * Write steps for generating tile wrappers.

Generation
''''''''''

1. The function generate_data_item is called, and determines if a new Tile Wrapper
   needs to be generated based on the details contained in the Task Function Specification.
   If it does, a new instance of TileWrapperGenerator is created, and the member
   functions `generate_header_code` and `generate_source_code` are called.

2. Since Tile Wrappers do not transfer data across devices, the code generator
   does not need to create as much setup code as the DataPacketGenerator. 

Data Packets
""""""""""""

DataPackets are used when data needs to be offloaded to a device. As such, a
DataPacket is responsible for determining the memory layout of all of the data
on the device, requesting that memory to be allocated, and copying that data to
pinned memory so the milhoja runtime can move it over to the device. Because DataPackets
need to allocate space for variables given in the Task Function Specification,
DataPacket generator will need an additional JSON containing the byte sizes for
every data type used in the DataPacket JSON input. This allows for the DataPacket
to be more memory efficient.

There are two principles that the DataPacket adheres to in order to improve
memory efficiency and speed.

1.  First, since the data packets are being used on a remote memory deivce, the way
    the information in the DataPacket is stored should be appropriate for that device.
    Given that these DataPacket classes are primarily for use with a remote GPU, the
    data is copied over using the Struct of Arrays (SoA) pattern. This is a performant
    pattern for memory being used with GPUs.

2.  Second, since the DataPackets are being created and managed at a very low level,
    it's important that the DataPacket uses its space efficiently. Because the
    DataPackets deal memory at the byte level, it's crucial to ensure that each
    section in the DataPacket is on a byte alignment boundary that matches the boundary
    of the hardware that the DataPacket and associated Task Function are being
    used on, to avoid memory alignment errors. For this reason, the `data_item` section
    of the Task Function Specification contains a "byte_align" attribute. This attribute
    inside of the specification also allows the generator to sort variables by size
    and add any necessary padding to each data array in the packet, further improving
    memory efficiency.

DataPacket Requirements
'''''''''''''''''''''''

1.  The DataPacket generator shall construct C++ classes derived from Milhoja's
    DataPacket base class and any other necessary files for use with either
    corresponding C++ or Fortran task functions, but not both.

2.  In order to generate a DataPacket, the DataPacket generator shall require an
    input data structure that is, or is derived from, :py:class:`milhoja.TaskFunction`,
    as well as an input file containing the sizes of every data type of every
    variable defined inside of the :py:class:`milhoja.TaskFunction` or its derived
    type, called the sizes json.

3.  The contents of an input data structure should not be dependent on the language of
    the corresponding task function or vice-versa. Thus, the information contained
    within the input data file shall be language agnostic.

    .. todo::
        * This might be a requirement of the TaskFunction class, not the DataPacket...

4.  The DataPacket generator shall attempt to optimize the derived DataPacket
    class such that it uses as little memory as possible, given the information
    from the data structure input and the sizes json.

5.  The DataPacket generator shall assume that the information in the given input
    is self-contained and doesn't require any external files, outside of those
    found in the Milhoja Runtime, for generated code to function properly.

6.  The DataPacket generator shall have a consistent set of rules, across all
    possible outputs, for mapping data inside of the input data structure to the 
    variables contained within the generated derived DataPacket class. It's crucial
    that variable names have consistent rulings, in the event that other code
    generators need to reference variables contained inside of the derived DataPacket
    class.

7.  For each variable defined inside of the input data structure, a generated
    derived DataPacket class will request that the runtime allocates space with
    appropriate padding for that variable onto the external device. The derived
    class will also create a pinned memory pointer and device memory pointer for
    each variable. The derived class is also responsible for copying the memory
    from the host variables into pinned memory, as well as copying pinned memory
    back to the host memory.

8.  For any variables contained inside of the input data structure that are considered
    :ref:`users_manual:external` variables, the generated derived DataPacket class
    will contain a third "host" pointer that contains the value of the variable when
    it was passed into the constructor.

Generation
''''''''''

In order to generate a DataPacket subclass, the code uses an external tool called
CG-Kit to simplify assembling the generated code into a file. This external tool
uses template files to assemble code into a main file by using keywords like _param,
_link, and _connector. The code that generates the DataPacket class is responsible
for assembling any necessary _params, _links, and _connectors, and calling CG-Kit
to combine the fragments into one or more code files for using the derived DataPacket
class.

The steps for generating a DataPacket subclass are as follows:

1.  The DataPacket generator takes in a Task Function Specification, destination,
    overwrite flag, a path to the Milhoja library installation using the generated
    code, indentation value, and a logging object derived from Milhoja.AbcLogger.

2.  The DataPacket generator loads each JSON and creates any extra information it
    needs for DataPacket class generation.

3.  The generator iterates through each variable in a specific internal order: [#]_

        1. external sources
        2. tile_metadata sources
        3. arrays that only have the 'variables_in' attribute. (read)
        4. arrays that have both the 'variabels_in' and 'variables_out' attribute. (read-write)
        5. arrays that only have the 'variables_out' attribute. (write)
        6. scratch data sources.
        7. creates new variables necessary for C++ Task Functions.

    A section inside of the DataPacket is generated for each item in the internal
    iteration order. The variables created in each section are sorted by size
    and formatted links, connectors, and param strings are created for later
    use with CG-Kit.

   a.   Using the information from the JSON input, the DataPacket generator will
        override methods from the base DataPacket class as necessary. The methods
        `extraAsynchronousQueue()` and `releaseExtraQueue()` are overriden if
        there are more than 0 n-extra-streams. The clone method is overridden,
        using the arguments from the `external` section to create a new packet,
        satisfying the prototype design pattern of the Data Items. The `unpack()`
        function uses information from read, read-write, and write sections
        to unpack the information from device memory back into host memory.

   b.   The `pack()` function is generated using three distinct phases. The first
        is the size determination phase, where the size of each item in the packet
        is determined, as well as the overall size of the packet. The next phase
        is the pointer determination phase. Since the packet uses a SoA pattern,
        the pointer determination phase gets the start of each array of pointers
        for each item in the packet. The number of tiles along with the size of 
        a given section determine the total size of data that the pointer points
        to. The last phase is the memory copy phase, where pointers from the host
        memory are copied into pinned memory for use by the runtime.

.. [#] Note that this is used internally only, argument order is preserved through
       :ref:`users_manual:argument_list`. 

4.  Once the generator has created each section each section in the JSON, it will
    write the generated strings for each section to CG-Kit template files. These
    files are called **cg-tpl.helper_{name}.cpp** and **cg-tpl.outer_{name}.cpp**,
    where **{name}** is the `data_item_class_name` found the the Task Function
    Specification api.

5.  Once the files have been generated, the DataPacket generator calls CG-Kit to 
    insert each param, connector, and link into the premade template files named
    **cg-tpl.datapacket.cpp** and **cg-tpl.datapacket_header.cpp** (included with
    the milhoja pypackage), resulting in the completed implementation of the new
    DataPacket class. The output file names are determined in the :ref:`users_manual:data_item`
    section.

.. todo::
    * The Data Item generators do not actually generated the inter-language code,
      they are generated after the task functions are generated.

6.  If the 'fortran' language is specified, the DataPacket generator will call two
    more functions to create interoperability layers for the new DataPacket. One
    is the C++ to C layer, and the other is the C to Fortran layer. The C to Fortran
    layer and the C++ to C layer are created using the same inputs used for the DataPacket.

7.  The C to Fortran layer generation creates a new Fortran 90 file that converts
    the C pointers and variable members in the DataPacket to Fortran based variables,
    then calls the Fortran task function associated with the generated DataPacket class. 
    The name of the file is determined from the Task Function Specification.

8.  The C++ to C layer is created using CG-Kit. Two more template files are generated
    and are combined with pre-existing template files to create the layer. The
    generated template file names are determined from the Task Function Specification
    and the existing templates are **cg-tpl.cpp2c_no_extra_queue.cpp** or **cg-tpl.cpp2c_extra_queue.cpp**
    and **cg-tpl.cpp2c.cpp**.

Data Mapping
^^^^^^^^^^^^

Every item specified in a JSON file will have a mapping in the data packet associated
with it. The data packet generator will create multiple variables for use within
and outside of the data packet. The variables shall be called the name of the associated
item surrounded by a prefix and a suffix.

For items in host memory, each item in the JSON will have an associated variable
in the data packet that starts with the prefix '_', followed by the name of the
item, followed by the suffix '_h'. Items contained in the 'external' section are the 
only variables contained in the data packet that have associated host variables
in the data packet. Example: 'dt' -> '_dt_h'.

For items in device memory, each item in the JSON will have an associated variable
in the data packet that starts with the prefix '_', followed by the name of the
item, followed by the suffix '_d'. Every item in the JSON will have an associated
device pointer. Example: 'dt' -> '_dt_d'. 

Items in the read and read-write sections have pinned memory pointers associated
with them in the data packet. This starts with the prefix '_', followed by the
name of the item, followed by the suffix '_p'. Example: 'Uin' -> '_Uin_p'.

When creating a packet using the 'cpp' language option, each item in read, read-write,
write, and scratch sections will have FArray{N}D device memory pointers associated
with them. The name of the pointer starts with the prefix '_f4_', followed by the
name of the item, followed by the suffix '_d'.

Code Generation Interface
-------------------------

Code Generation classes contained inside of the code generation interface follow
a loose naming pattern that determines what language and device the code is generated
for. The general format is as follows:

`[object][_device][_language]`

Where the `[object]` is the type of object being generated, either `DataPacket`
or `TileWrapper`. The `[_device]` is the device that the generated code will run
on, either `cpu` for running on the cpu, or the tpye of offloading to use if the
gpu is being used. Finally, the `[_language]` is the language that the code is
generated for. For example, `TaskFunctionC2FGenerator_OpenACC_F` is a class
for generating the C2F layer for TaskFunctions, with code that runs on the GPU
with OpenACC offloading, generated for fortran code.

Since this is just a 'loose' rule, there are a few exceptions to this. The
TileWrapperGenerator class and the DataPacketGenerator class hide any language specific
generation rules inside of themselves, so there are no separate generators for language
or device specific data items. Generally, TileWrappers are only used on CPUs, so
there is no need to specify the device. TileWrappers are also compatible with both
Fortran & C++ Task Functions without extra modifications. DataPackets are designed
in a similar way, however, the generated code is not necessarily compatible with
both C++ and Fortran Task Functions (this will likely change in the future), and
DataPackets are primarily used for offloading purposes. The last exceptions are the
`DataPacketModGenerator` and the `TileWrapperModGenerator`, which exist to generate
fortran module interface files for use with the TaskFunction, therefore there is
no need to specify the language or device/offloading inside of the file name.

.. autoclass:: milhoja.TileWrapperGenerator
    :members:
.. autoclass:: milhoja.TileWrapperModGenerator
    :members:
.. autoclass:: milhoja.DataPacketGenerator
    :members:
.. autoclass:: milhoja.TaskFunctionGenerator_cpu_cpp
    :members:
.. autoclass:: milhoja.TaskFunctionGenerator_cpu_F
    :members:
.. autoclass:: milhoja.TaskFunctionGenerator_OpenACC_F
    :members:
.. autoclass:: milhoja.TaskFunctionC2FGenerator_cpu_F
    :members:
.. autoclass:: milhoja.TaskFunctionC2FGenerator_OpenACC_F
    :members:
.. autoclass:: milhoja.TaskFunctionCpp2CGenerator_cpu_F
    :members:
.. autoclass:: milhoja.TaskFunctionCpp2CGenerator_OpenACC_F
    :members:
