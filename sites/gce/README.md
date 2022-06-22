The contents of this folder are used for developing Milhoja on different GCE machines and for testing with the GCE CI system.

Milhoja gatekeepers maintain in the GCE filesystem for the purposes of CI testing files for setting up software stacks as well as Milhoja dependencies built with these stacks.  This installation is located at
```
/nfs/gce/projects/Milhoja/MilhojaTest/gce
```
Note that while this is a repository, it should not be used as such.  In particular, it should not be modified or pulled.  Care should be taken to use the contents of the repo beyond what is given here.

Users can load a SW stack by sourcing scripts with names that follow the naming convention `setup_current_<stack name>_stack.sh`.  The Milhoja dependencies built with these stacks are in the folders that follow the naming convention `<compiler family>_current`.

To use the Makefiles in this folder without any need to alter them,
* set the environment variable `MILHOJA_CODE_REPO` to the location of you Milhoja code repository and
* set the environment variable `MILHOJA_TEST_REPO` to `/nfs/gce/projects/Milhoja/MilhojaTest`.

Please note that running on GCE is complicated by the fact that the machines use different hardware and OS.  Therefore, use of the contents of this folder or the folder in the GCE installation is not mindless plug-and-play.
