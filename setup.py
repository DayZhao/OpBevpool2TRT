from setuptools import setup
import cpp_extension
from torch.utils.cpp_extension import BuildExtension

setup(name='bevpool_ext', 
      ext_modules=[cpp_extension.CppExtension('bevpool_ext', ['/ops/bevpool/src/bev_pool.cpp', '/ops/bevpool/src/bev_pool_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension}
)
