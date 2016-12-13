# Copyright 2016 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import numpy
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from os.path import join


long_desc = 'SparseArray'
version = open("VERSION").readline().lstrip().rstrip()
lst = open(join("SparseArray", "__init__.py")).readlines()
for k in range(len(lst)):
    v = lst[k]
    if v.count("__version__"):
        lst[k] = "__version__ = '%s'\n" % version
with open(join("SparseArray", "__init__.py"), "w") as fpt:
    fpt.write("".join(lst))

extension = [Extension('SparseArray.sparse_array', ["SparseArray/sparse_array.pyx"],
                       define_macros=[('CYTHON_TRACE', '1'),
                                      ('CYTHON_TRACE_NOGIL', '1')])]

setup(
    name="SparseArray",
    description="""SparseArray""",
    long_description=long_desc,
    version=version,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    url='https://github.com/mgraffg',
    author="Mario Graff",
    author_email="mgraffg@ieee.org",
    ext_modules=cythonize(extension,
                          compiler_directives={'profile': True,
                                               'linetrace': True,
                                               'nonecheck': False,
                                               'binding': True,
                                               'embedsignature': True,
                                               'boundscheck': False}),
    packages=['SparseArray', 'SparseArray/tests'],
    include_package_data=True,
    zip_safe=False,
    package_data={'': ['*.pxd']},
    install_requires=['cython >= 0.19.2']
)
