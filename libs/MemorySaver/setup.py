from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name='MemorySaver',
    ext_modules=[
        CUDAExtension(
            name = 'MemorySaver',
            sources = [
                'src/MemorySaver_cuda.cpp',
                'src/MemorySaver_kernel_forward.cu',
                'src/MemorySaver_kernel_max_avg_argmax_backward.cu',
                'src/MemorySaver_kernel_compression_backward.cu',
                ],
            extra_compile_args={
                'cxx':[
                ],
                'nvcc':[
                ]},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
