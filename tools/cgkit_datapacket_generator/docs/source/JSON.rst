JSON
====

| There are several sections that can be used inside of the JSON file. Most are optional, but some are not. 
| Default values for some items in each section will be specified in [brackets]. Otherwise, the item is required to be in the JSON file.

Packet Metadata
---------------

| These options in the JSON file appear outside of any section and exist to help the generator with general packet information. 
| These are all possible parameters for the packet metadata:

* **byte-align**: The specified byte alignment. [16]
* **n-extra-streams**: The number of extra streams to use. [0]
* **task-function-argument-list**: The order of arguments for the task function. This is required since JSONs are unordered.

constructor
-----------
Non-tile-specific data goes here. Note that the number of tiles is automatically inserted into the data packet, 
so those do not need to be specified in the JSON file. Any items in general must be passed into the data packet 
constructor, and the data packet will take ownership of the items and copy them into its own memory to manage it. 
We are assuming the items will remain valid throughout the life cycle of the entire packet. This section is 
generally good for non-tile specific variables that are used in your computation functions.

* **dt: Real**: The delta time.

tile-metadata
-------------
Tile-metadata is a dictionary containing metadata for each tile. The items in tile-metadata consist of a key, 
a specifier that may be referenced by items in the various array sections, and the value is an obtainable value 
from the Tile class. The list of obtainable values is a set of keywords contained in the data packet generator code. 

tile-in
-------

tile-in-out
-----------

tile-out
--------

tile-scratch
------------