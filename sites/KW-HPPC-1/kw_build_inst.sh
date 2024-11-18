# Buid a Milhoja-ish library and install it in my private space - kw
#
# Modify manually in 2 places below to build for 1D or 3D instead of 2D.

export MILHOJA_CODE_REPO=/home/kweide/projects/Milhoja
export KW_INSTALL_DIR=/home/kweide/projects/local-milhoja/2d

set -xve
cd ${MILHOJA_CODE_REPO} # top of cloned Milhoja_repo

test -f Makefile.configure   && rm -v  Makefile.configure
test -d build/               && rm -rv build/
test -d "${KW_INSTALL_DIR}"  && rm -rv "${KW_INSTALL_DIR}"
./configure.py --makefile $PWD/sites/KW-HPPC-1/Makefile.site.gnu_openmpi --dim 2 --runtime HOSTMEM --grid None --offload OpenACC --prefix ${KW_INSTALL_DIR} \
	       --support_push --support_packets
make clean
# The following is required on gce for building sizes.json, only needed for --support_packets
export MILHOJA_TEST_CLONE=/nfs/gce/projects/Milhoja/MilhojaTest
export JSON_CODE_REPO=/home/kweide/projects/json
make -j2 all
make install
