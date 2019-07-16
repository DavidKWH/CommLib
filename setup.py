from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# define extension modules
min_sum_decoder = Extension('dh_comm.ldpc.ldpc_decoder_oms',
                   sources=['dh_comm/ldpc/ldpc_decoder_oms.pyx',
                            'dh_comm/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','1')],
                   libraries=['m']) # -lm
bp_decoder      = Extension('dh_comm.ldpc.ldpc_decoder_bp',
                   sources=['dh_comm/ldpc/ldpc_decoder_bp.pyx',
                            'dh_comm/ldpc/layered_ldpc_dec.c'],
                   define_macros=[('USE_CNODE_FCN','2')],
                   libraries=['m']) # -lm

extensions = [min_sum_decoder, bp_decoder]

setup(name='dh_comm',
      version='0.1.0',

      packages=find_packages(),
      package_data={'': ['*.txt']},
      install_requires=["numpy",
                        "Cython",
                        "importlib_resources ; python_version<'3.7'"],
      ext_modules = cythonize(extensions, language_level=3),

      description='Custom communication package',
      author='David K. W. Ho',
      author_email='davidkwho@gmail.com',
      #url='',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: POSIX :: Linux",
      ],
      )
