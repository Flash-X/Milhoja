How to use the DataPacket Generator
===================================

Using the DataPacket Generator as a standalone tool does require a bit of setup. The generator employs the use of 
CG-Kit, a code generation tool kit. CG-Kit can be installed from https://github.com/johannrudi/cg-kit by following 
the installation instructions (Use develop mode?). You also may need to adjust your python version in order to install 
it. On summit, for example, you may need to switch to the python 3.8 module with anaconda for easy installation of cgkit. 
Or, you can create a python virtual environment and install it. 

OR OR, if you are on summit you can use summit's module system to load the necessary modules for installing cgkit. 


Once CG-Kit is installed, the DataPacket Generator can now be used as a standalone tool. 

To setup the files necessary for the generator, you will need to be aware of a few file paths that are not passed 
in via command line. These files are the various CG-Kit template files that are used to build the DataPacket 
subclasses.

The full list of files needed are:

* cg-tpl.datapacket_header.cpp
* cg-tpl.cpp2c.cpp 
* cg-tpl.cpp2c_extra_queue.cpp
* cg-tpl.cpp2c_no_extra_queue.cpp
* cg-tpl.datapacket.cpp

These are all of the template files that the DataPacket Generator uses to assemble the finished DataPacket subclass, 
along with files that are generated when the script is ran. These files are assumed to be in the same folder that the 
generator is contained within. Generated packets are not guaranteed to work properly if these files are modified. 
These files do not require any setup, only that they are placed within the same folder at the DataPacket Generator. 

In order to run the DataPacket Generator, use this command:

`./generate_packet.py -l [fortran | cpp] -s [/path/to/sizes.json] /path/to/datapacket_generator_input.json`

Where -l is the language of the task function to generate for, -s is the path to the sizes.json file, and the final 
input is the path to the JSON file that contains the necessary format and information for generating a DataPacket 
subclass. Currently, the only appropriate sizes file that exists is meant for summit, and that is the sizes file 
that exists in the branch in the sample_jsons folder. Providing a sizes file is not necessary, but the packet is 
not guaranteed to work correctly if the sizes of each datatype are not specified. 

A few example commands to run...

* `./generate_packet.py -l cpp -s sample_jsons/sizes.json sample_jsons/DataPacket_Hydro_gpu_3.json`

	* This command will generate the 3rd variant of the Hydro DataPacket for use with a C++ task function, using 
		sizes.json as the list of byte sizes.
		
* `./generate_packet.py -l fortran -s sample_jsons/sizes.json sample_jsons/DataPacket_Hydro_gpu_3.json`

	* This command will generate the 3rd variant of the Hydro DataPacket for use with a fortran task function,
		with sizes.json as the list of byte sizes. This command will also generate the cpp2c and c2f interoperability 
		layers for the packet, allowing it to be used with a fortran task function of the same name. (Currently only
		works with Sedov problem). 

* `./generate_packet.py -l cpp -s sample_jsons/sizes.json sample_jsons/DataPacket_Hydro_gpu_2.json`

	* This command will generate the 2nd variant of the Hydro DataPacket for use with a C++ task function.
	 
A few things to note when generating packets:

	* The output of the DataPacket Generator is as follows:

		* cgkit.datapacket.h -> The header file of the assembled data packet
		* cgkit.datapacket.cpp -> The file containing the packet code
		* cgkit.cpp2c.cxx -> The file containing the cpp2c interoperability layer
		* c2f.F90 -> The file containing the c2f interoperability layer

	* cgkit.cpp2c.cxx and c2f.F90 will not be created when generating a DataPacket subclass for use with 
		a C++ task function.










HERE ARE MY NOTES ON TRYING TO SETUP CGKIT FOR SUMMIT WITHOUT THE USE OF ANACONDA

I started by creating an empty virtual environment. Tried to install the most up to date versions of the packages that cg-kit needs to no avail.
I tried installing the versions that CG-Kit specifies to no avail. Keep running into the same issues with numpy, either Cythonize fails, 
or MATHLIB env variable is not set. Cython is installed and its the correct one that CG-Kit uses so I'm not sure why the cythonizing is failing 
for installing numpy. Will continue troubleshooting tomorrow. 
