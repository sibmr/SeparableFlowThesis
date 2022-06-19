from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name='MemorySaver',
    ext_modules=[
        CUDAExtension(
            name = 'MemorySaver', 
            sources = ['src/MemorySaver_cuda.cpp', 'src/MemorySaver_kernel.cu',],
            # NOTE: try 'nvcc':['-O3']
            extra_compile_args={'cxx':[],'nvcc':[]},            
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
