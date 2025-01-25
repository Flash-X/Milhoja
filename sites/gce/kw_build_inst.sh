# Buid a Milhoja-ish library and install it in my private space - kw
#
# Modify manually in 2 places below to build for 1D or 3D instead of 2D.

export MILHOJA_CODE_REPO=/home/kweide/projects/Milhoja
export KW_INSTALL_DIR=/nfs/gce/projects/FLASH5/kweide/local-milhoja/2d

set -xve
cd ${MILHOJA_CODE_REPO} # top of cloned OrchestrationRuntime repo

test -f Makefile.configure   && rm -v  Makefile.configure
test -d build/               && rm -rv build/
test -d "${KW_INSTALL_DIR}"  && rm -rv "${KW_INSTALL_DIR}"
./configure.py --makefile $PWD/sites/gce/Makefile.site.gnu_mpich --dim 2 --runtime HOSTMEM --offload OpenACC --prefix ${KW_INSTALL_DIR} \
	       --support_exec --support_push --support_packets
make clean
# export MILHOJA_TEST_CLONE=/nfs/gce/projects/Milhoja/MilhojaTest
make -j12 all
make install
