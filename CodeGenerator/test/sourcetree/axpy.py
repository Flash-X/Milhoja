import pathlib, sys
sys.path.append('../../')
from codeflow.assembler.sourcetree import SourceTree

SOURCETREE_OPTIONS = {
    'codePath': pathlib.Path('.'),
    'linkConnectorKeys': ['main', 'setup', 'execute', 'kernel'],
    'indentSpace': ' '*2,
    'verbose': True
}

LINES_REF_MAIN = \
'''
/* <_connector:main> */
/* <_link:setup> */
/* </_link:setup> */
int main(void)
{
  /* ... initialize ... */
  /* <_link:execute> */
  /* </_link:execute> */
  /* ... finalize ... */
}
/* </_connector:main> */
'''

LINES_REF_GPU = \
'''
/* <_connector:setup> */
__global__
void _param:axyFunction(_param:axyType _param:a, _param:axyType *_param:x, _param:axyType *_param:y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  /* <_link:kernel> */
  /* </_link:kernel> */
}
/* </_connector:setup> */
/* <_connector:execute> */
_param:axyFunction<<<_param:size/320, 320>>>(_param:a, _param:x, _param:y);
/* </_connector:execute> */
'''

LINES_REF_CPU = \
'''
/* <_connector:execute> */
for (int i=0; i<_param:size; i++) {
  /* <_link:kernel> */
  /* </_link:kernel> */
}
/* </_connector:execute> */
'''

LINES_REF_KERNEL = \
'''
/* <_connector:kernel> */
_param:y[_param:idx] = _param:a*_param:x[_param:idx] + _param:y[_param:idx];
/* </_connector:kernel> */
'''

LINES_REF_ASSEMBLED_GPU = \
'''
/* <_connector:main> */
/* <_link:setup> */
/* <axpy_gpu.suffix> */
__global__
void saxpy(float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  /* <_link:kernel> */
  /* <axpy_kernel.suffix> */
  y[i] = a*x[i] + y[i];
  /* </axpy_kernel.suffix> */
  /* </_link:kernel> */
}
/* </axpy_gpu.suffix> */
/* </_link:setup> */
int main(void)
{
  /* ... initialize ... */
  /* <_link:execute> */
  /* <axpy_gpu.suffix> */
  saxpy<<<3200000/320, 320>>>(a, x, y);
  /* </axpy_gpu.suffix> */
  /* </_link:execute> */
  /* ... finalize ... */
}
/* </_connector:main> */
'''

LINES_REF_ASSEMBLED_CPU = \
'''
/* <_connector:main> */
/* <_link:setup> */
/* </_link:setup> */
int main(void)
{
  /* ... initialize ... */
  /* <_link:execute> */
  /* <axpy_cpu.suffix> */
  for (int i=0; i<3200000; i++) {
    /* <_link:kernel> */
    /* <axpy_kernel.suffix> */
    y[i] = a*x[i] + y[i];
    /* </axpy_kernel.suffix> */
    /* </_link:kernel> */
  }
  /* </axpy_cpu.suffix> */
  /* </_link:execute> */
  /* ... finalize ... */
}
/* </_connector:main> */
'''

def perform_check_on_file(file_chk, lines_ref, printLines=False):
    assembler = SourceTree(**SOURCETREE_OPTIONS)
    assembler.initialize(file_chk)
    lines_chk = assembler.parse()
    if printLines:
        print(lines_chk)
    print('Check "parse {}" \t {}'.format(file_chk, lines_ref.strip() == lines_chk.strip()))

def perform_check_on_assembled_gpu(suffix, printLines=False):
    assembler = SourceTree(**SOURCETREE_OPTIONS)
    assembler.initialize('axpy_main'+suffix)
    linkLoc = ('_connector:main', )
    loc     = assembler.link('axpy_gpu'+suffix, linkLoc)
    linkLoc = tuple([*linkLoc, *loc[0]])
    loc     = assembler.link('axpy_kernel'+suffix, linkLoc)
    lines_chk = assembler.parse()
    if printLines:
        print(lines_chk)
    lines_ref = LINES_REF_ASSEMBLED_GPU.replace('.suffix', suffix)
    print('Check "parse assembled GPU code from {} files" \t {}'.format(suffix, lines_ref.strip() == lines_chk.strip()))

def perform_check_on_assembled_cpu(suffix, printLines=False):
    assembler = SourceTree(**SOURCETREE_OPTIONS)
    assembler.initialize('axpy_main'+suffix)
    linkLoc = ('_connector:main', )
    loc     = assembler.link('axpy_cpu'+suffix, linkLoc)
    linkLoc = tuple([*linkLoc, *loc[0]])
    loc     = assembler.link('axpy_kernel'+suffix, linkLoc)
    lines_chk = assembler.parse()
    if printLines:
        print(lines_chk)
    lines_ref = LINES_REF_ASSEMBLED_CPU.replace('.suffix', suffix)
    print('Check "parse assembled CPU code from {} files" \t {}'.format(suffix, lines_ref.strip() == lines_chk.strip()))

def main():
    perform_check_on_file('axpy_main.json', LINES_REF_MAIN)
    perform_check_on_file('axpy_main.cpp', LINES_REF_MAIN)

    perform_check_on_file('axpy_gpu.json', LINES_REF_GPU)
    perform_check_on_file('axpy_gpu.cpp', LINES_REF_GPU)

    perform_check_on_file('axpy_cpu.json', LINES_REF_CPU)
    perform_check_on_file('axpy_cpu.cpp', LINES_REF_CPU)

    perform_check_on_file('axpy_kernel.json', LINES_REF_KERNEL)
    perform_check_on_file('axpy_kernel.cpp', LINES_REF_KERNEL)

    perform_check_on_assembled_gpu('.json')
    perform_check_on_assembled_gpu('.cpp')

    perform_check_on_assembled_cpu('.json')
    perform_check_on_assembled_cpu('.cpp')

#TODO compare trees

if '__main__' == __name__:
    main()
