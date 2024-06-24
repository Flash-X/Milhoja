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

.. todo::
  * Fix broken links.

There are two major types of codes that the Milhoja pypackage can generate for the
user when provided with a `TaskFunction Specification <#>`_. These are the `TaskFunction <#Task Functions>`_
and the `Data Items <#Data Items>`_. These two types of codes are considered a pair,
so one JSON input should be used to generate both at the same time. One is not
guaranteed to work without the other. Ultimately, the application using this package
decides what code needs to be generated.

Task Functions
^^^^^^^^^^^^^^

Task function generation doc goes here?

.. todo::
  * Write task function overview.
  * Write steps for generating task functions.

Data Items
^^^^^^^^^^

Generated Data Items are responsible for holding the information needed by the
Task Function, and work in tandem with the Milhoja runtime if using any device
offloading. There are two types of Data Items, `Tile Wrappers <#Tile Wrappers>`_
and `Data Packets <#Data Packets>`_.

Tile Wrappers
"""""""""""""

Tile Wrappers are used for data that does not need to be offloaded.

.. todo::
  * Write tile wrapper overview.
  * Write steps for generating tile wrappers.

Data Packets
""""""""""""

Whether or not a DataPacket subclass needs to be generated is the responsiblity of the application that is using Milhoja and 
its code generation tools. If a new DataPacket subclass needs to be generated, the DataPacket code generator will be used.
When the generator is called, it will create new files based on the information passed to it. To do this, the DataPacket 
generator will need:

* The byte sizes for every data type used in the DataPacket JSON input.
* The language that is being used to run the task function. Either 'cpp' or 'fortran'.
* A JSON input containing all information needed for each item in a data packet. 
  For specifications on this, read the DataPacket JSON section in the UserManual.

Since an important aspect of the DataPacket is that it's as efficient as possible, there are a number of things to be considered 
when generating a new DataPacket subclass. 

First, since the data packets are being used on a remote memory deivce, the way the information in the DataPacket is stored 
should be appropriate for that device. Given that these DataPacket classes are mostly for use with a remote GPU, the data is 
copied over using the Struct of Arrays (SoA) pattern. This is a performant pattern for memory being used with GPUs. It's also 
used to overcome certain challenges in generating Fortran task functions. In order to effectively generate code following the 
SoA pattern, the DataPacket JSON defines various sections for specifying variables. For more information, see the 
input_interface/DataPacket JSON section in the UserManual.

Second, since the DataPackets are being created and managed a very low level, it's important that the DataPacket uses its space 
efficiently. Because the DataPackets deal memory at the byte level, it's crucial to ensure that each section in the DataPacket 
is on a byte alignment boundary that matches the boundary of the hardware that the DataPacket and associated task function 
are being used on, to avoid memory alignment errors. For these reasons, we force the user to specify a byte-alignment value. 
For similar reasons, the DataPacket generator requires the byte sizes of each variable in the data packet. This is so that the 
variables can be sorted inside of the packet from largest to smallest, potentially reducing the total amount of necessary 
padding and being more performant.

Using that information, the DataPacket generator can create a new subclass for passing information to a remote device. 

Packet Generation Steps
"""""""""""""""""""""""

In order to generate a DataPacket subclass, the code uses an external tool called CG-Kit to simplify assembling the generated 
code into a file (TODO: will CG-Kit be discussed in the developer's guide in depth?). This external tool uses template files 
to assemble code into a main file by using keywords like _param, _link, and _connector. The code that generates the DataPacket 
class is responsible for assembling any necessary _params, _links, and _connectors, and calling CG-Kit to combine the fragments 
into one or more code files for using the derived DataPacket class.

The steps for generating a DataPacket subclass are as follows: 

1. The DataPacket generator takes in a language specifier, a byte sizes JSON, and DataPacket. 

2. The DataPacket generator loads each JSON and creates any extra information it needs for DataPacket class generation.

3. The generator moves through each section in the DataPacket JSON in a specified order [`constructor`, `tile_metadata`, `tile_in`, 
   `tile_in_out`, `tile_out`, and `tile_scratch`], sorts the section by size, then generates formatted links, connectors, 
   and param strings and stores them in dictionaries for later use. The links, connectors, and params from the DataPacket 
   generator save the implementation for the DataPacket. 

   a. Using the information from the JSON input, the DataPacket generator will implement methods from the base DataPacket class as necessary.
      The methods `extraAsynchronousQueue()` and `releaseExtraQueue()` are given derived class implementations if there are more than 
      0 n-extra-streams. The clone method is overridden, using the arguments from the `constructor` section to create a new packet, 
      satisfying the prototype design pattern. The `unpack()` function uses information from `tile_in`, `tile_in_out`, and `tile_out` to 
      unpack the information from device memory back into host memory.

   b. The `pack()` function is generated using three distinct phases. The first is the size determination phase, where the size of each item 
      in the packet is determined, as well as the overall size of the packet. The next phase is the pointer determination phase. Since the 
      packet uses a SoA pattern, the pointer determination phase gets the start of each array of pointers for each item in the packet. The 
      size of each item's pointer array is equal to the number of tiles. The last phase is the memcpy phase, where pointers from the host 
      memory are copied into pinned memory.

4. Once the generator has gone through each section in the JSON, it will write the strings in the dictionaries to CG-Kit 
   template files for use by CG-Kit. These files are **cg-tpl.datapacket_helpers.cpp** and **cg-tpl.datapacket_outer.cpp**.

5. Once the files have been generated, the DataPacket generator calls CG-Kit to insert each param, connector, and link into 
   the premade template files named **cg-tpl.datapacket.cpp** and **cg-tpl.datapacket_header.cpp**, resulting in the completed
   implementation of the new DataPacket class. The output files are named **cgkit.[name].cpp** and **cgkit.[name].h**, where
   [name] is the basename of whatever file was used to generate the packets, without the file extention. 

6. If the 'fortran' language is specified, the DataPacket generator will call two more functions to create interoperability 
   layers for the new DataPacket. One is the C++ to C layer, and the other is the C to Fortran layer. The C to Fortran layer and 
   the C++ to C layer are created using the same inputs used for the DataPacket.

7. The C to Fortran layer generation creates a new Fortran 90 file that converts the C pointers and variable members in the 
   DataPacket to Fortran based variables, then calls the Fortran task function associated with the generated DataPacket class. 
   This file is named **c2f.f90**.
   
   a. For information on the C to Fortran layer, see :doc:`f2c`.

8. The C++ to C layer is created using CG-Kit. Two more template files are generated and are combined with pre-existing template 
   files to create the layer. The generated template files are named **cg-tpl.cpp2c_outer** and **cg-tpl.cpp2c_helper.cpp** and 
   the existing templates are **cg-tpl.cpp2c_no_extra_queue.cpp** or **cg-tpl.cpp2c_extra_queue.cpp** and **cg-tpl.cpp2c.cpp**.

   a. For more information of the C++ to C interoperability layer, see :doc:`f2c`.

Data Mapping
^^^^^^^^^^^^

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


Code Generation Interface
-------------------------
.. autoclass:: milhoja.TileWrapperGenerator_cpp
    :members:
.. autoclass:: milhoja.TaskFunctionGenerator_cpu_cpp
    :members:
