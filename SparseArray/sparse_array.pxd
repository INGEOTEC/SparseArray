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

from cpython cimport array
from cpython cimport list
cimport cython
from libc cimport math
ctypedef double (*two_arguments)(double, double)
ctypedef double (*one_argument)(double)


cdef inline double non_op(double a):
    return a


cdef inline double add_op(double a, double b):
    return a + b


cdef inline double sub_op(double a, double b):
    return a - b


cdef inline double minus_op(double a):
    return -a


cdef inline double zero_min_op(double a):
    return math.fmin(a, 0)


cdef inline double zero_max_op(double a):
    return math.fmax(a, 0)


cdef inline double div_left_op(double a):
    return math.INFINITY


cdef inline double div_right_op(double a):
    return 0


cdef inline double atan2_left_op(double a):
    return math.atan2(a, 0)


cdef inline double atan2_right_op(double a):
    return math.atan2(0, a)


@cython.cdivision(True)
cdef inline double div_op(double a, double b):
    return a / b


cdef inline double sq_op(double a):
    return a * a


cdef inline double sign_op(double a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    return 0


cdef inline double finite_op(double a):
    if math.isfinite(a):
        return a
    return 0


cdef inline double mul_op(double a, double b):
    return a * b


cdef inline double boundaries_op(double a):
    if a > 1:
        return 1
    elif a < -1:
        return -1
    return a


cdef inline double pow_left_op(double a):
    return 1.0


cdef inline double pow_right_op(double a):
    return 0.0


cdef inline void set_value(unsigned int *output_index,
                           double *output_data,
                           Py_ssize_t *c, unsigned int k,
                           double x):
    if x != 0:
        output_data[c[0]] = x
        output_index[c[0]] = k
        c[0] = c[0] + 1


cdef class SparseArray:
    cdef public unsigned int non_zero
    cdef unsigned int _len
    cdef public array.array index
    cdef public array.array data
    cdef void _empty(self, unsigned int len, unsigned int non_zero)
    cdef void fix_size(self, unsigned int new_size)
    cdef double getitem(self, unsigned int k)
    cdef unsigned int union_size(self, SparseArray second)
    cdef SparseArray union_func(self, two_arguments func,
                                one_argument left,
                                one_argument right,
                                SparseArray second)
    cpdef SparseArray add(self, SparseArray second)
    cpdef SparseArray add2(self, double second)    
    cpdef SparseArray sub(self, SparseArray second)
    cpdef SparseArray min(self, SparseArray second)
    cpdef SparseArray max(self, SparseArray second)
    cpdef SparseArray div(self, SparseArray second)
    cdef SparseArray one_argument_func(self, one_argument func, bint full)
    
    cpdef SparseArray acos(self)
    cpdef SparseArray asin(self)
    cpdef SparseArray atan(self)
    cpdef SparseArray atan2(self, SparseArray second)
    cpdef SparseArray cos(self)
    cpdef SparseArray sin(self)
    cpdef SparseArray tan(self)

    cpdef SparseArray cosh(self)
    cpdef SparseArray sinh(self)
    cpdef SparseArray tanh(self)
    cpdef SparseArray acosh(self)
    cpdef SparseArray asinh(self)
    cpdef SparseArray atanh(self)

    cpdef SparseArray hypot(self, SparseArray second)
    
    cpdef SparseArray exp(self)
    cpdef SparseArray expm1(self)
    cpdef SparseArray log(self)
    cpdef SparseArray log2(self)
    cpdef SparseArray log10(self)
    cpdef SparseArray log1p(self)
    
    cpdef SparseArray lgamma(self)

    cpdef SparseArray pow(self, SparseArray second)
    cpdef SparseArray sqrt(self)
    cpdef SparseArray sq(self)
    cpdef SparseArray sign(self)

    cpdef SparseArray fabs(self)
    cpdef SparseArray ceil(self)
    cpdef SparseArray floor(self)
    cpdef SparseArray trunc(self)

    cpdef SparseArray erf(self)
    cpdef SparseArray erfc(self)
    
    cpdef SparseArray finite(self, bint inplace=?)
    cpdef SparseArray boundaries(self)
    cpdef SparseArray copy(self)


    cdef unsigned int intersection_size(self, SparseArray second)
    cdef SparseArray intersection_func(self, two_arguments func,
                                       SparseArray second)
    cdef SparseArray full_func(self, two_arguments func,
                               one_argument left,
                               one_argument right,                               
                               SparseArray second)
    cpdef SparseArray mul(self, SparseArray second)
    cpdef SparseArray mul2(self, double second)
    cpdef double dot(self, SparseArray second)

    cpdef double sum(self)
    cpdef SparseArray unit_vector(self)
    cpdef double pearson_coefficient(self, SparseArray second)
    cpdef double cosine_distance(self, SparseArray second)
    cpdef double SSE(self, SparseArray second)
    cpdef double SAE(self, SparseArray second)
    
    cpdef bint isfinite(self)
