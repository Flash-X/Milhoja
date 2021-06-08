#!/usr/bin/env python3

### Fill Language-Independent Kernel Cores for Multiple Environments ###

import sys
from string import Template
import toml
import re
import os.path

class MyTemplate(Template):
    delimiter = '$'
    flags = re.VERBOSE
    idpattern = r'(?a:[_A-Za-z][_A-Za-z0-9]*)|[&]'
#    braceidpattern = r'(?a:[_A-Za-z][_A-Za-z0-9]*)'


if len(sys.argv) != 3:
    print("\n%s: %s\n%s\n" % (os.path.basename(sys.argv[0]),"Fill a kernel core template with definitions from a dictionary file,",
                                                          "               writing resulting text to standard output."))
    print("Usage:\n   %s %s %s\n" % (sys.argv[0],"template_file","dictionary_file"))
    print("For example:\n   %s %s %s\n" % (sys.argv[0],"hy_updateVelxHll.kernelcore","hy_oacc_summit.F90.toml"))
    sys.exit()

# Read the dictionary of replacement texts first
dictfn = sys.argv[2]
repl_dict = toml.load(dictfn)   # Could easily use other format instead of TOML

# Now read the template file text into memory
infn = sys.argv[1]
f = open(infn,'r')
kctext = f.read()
f.close()

# Replace ${key} or $key by the proper replacement text (if found)
tpl = MyTemplate(kctext)
txt1 = tpl.safe_substitute(repl_dict)

# Print to standard output
print(txt1)
