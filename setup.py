from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

# define extension modules
decoder_r1_2 = Extension('comm_ai.ldpc.ldpc_decoder_r1_2',
                   sources=['comm_ai/ldpc/ldpc_decoder_r1_2.pyx',
                            'comm_ai/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','1'),
                                  ('CODE_RATE','1')],
                   libraries=['m']) # -lm
decoder_r5_8 = Extension('comm_ai.ldpc.ldpc_decoder_r5_8',
                   sources=['comm_ai/ldpc/ldpc_decoder_r5_8.pyx',
                            'comm_ai/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','1'),
                                  ('CODE_RATE','2')],
                   libraries=['m']) # -lm
decoder_r3_4 = Extension('comm_ai.ldpc.ldpc_decoder_r3_4',
                   sources=['comm_ai/ldpc/ldpc_decoder_r3_4.pyx',
                            'comm_ai/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','1'),
                                  ('CODE_RATE','3')],
                   libraries=['m']) # -lm
decoder_r13_16 = Extension('comm_ai.ldpc.ldpc_decoder_r13_16',
                   sources=['comm_ai/ldpc/ldpc_decoder_r13_16.pyx',
                            'comm_ai/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','1'),
                                  ('CODE_RATE','4')],
                   libraries=['m']) # -lm
#min_sum_decoder = Extension('comm_ai.ldpc.ldpc_decoder_oms',
#                   sources=['comm_ai/ldpc/ldpc_decoder_oms.pyx',
#                            'comm_ai/ldpc/layered_ldpc_dec.c'],
#                   define_macros=[('USE_CNODE_FCN','1')],
#                   libraries=['m']) # -lm
#bp_decoder      = Extension('comm_ai.ldpc.ldpc_decoder_bp',
#                   sources=['comm_ai/ldpc/ldpc_decoder_bp.pyx',
#                            'comm_ai/ldpc/layered_ldpc_dec.c'],
#                   define_macros=[('USE_CNODE_FCN','2')],
#                   libraries=['m']) # -lm

extensions = [decoder_r1_2, decoder_r5_8, decoder_r3_4, decoder_r13_16]
#extensions = [min_sum_decoder, bp_decoder]

setup(name='comm_ai',
      version='0.2.0',
      packages=find_packages(),
      package_data={'': ['*.txt']},
      scripts=['bin/rclone.py'],
      install_requires=["numpy",
                        "Cython",
                        "redis",
                        "psutil",
                        "importlib_resources ; python_version<'3.7'"],
      ext_modules = cythonize(extensions, language_level=3),
      include_dirs=[numpy.get_include()],
      #license='LICENSE.txt',
      description='Physical layer communications package',
      #long_description=open('README.txt').read(),
      #url='http://pypi.python.org/pypi/CommLib/',
      author='David K. W. Ho',
      author_email='davidkwho@gmail.com',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: POSIX :: Linux",
      ],
      )
