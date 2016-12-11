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
# cython: profile=True
# cython: nonecheck=False


cimport cython
import types
from cpython cimport array


cpdef rebuild(data, index, size):
    cdef SparseArray r = SparseArray()
    cdef Py_ssize_t k = 0
    r.empty(size, len(data))
    for d, j in zip(data, index):
        r.data[k] = d
        r.index[k] = j
        i += 1
    return r


@cython.freelist(512)
cdef class SparseArray:
    def __cinit__(self):
        self.non_zero = 0
        self._len = 0
        self.index = array.array('I')
        self.data = array.array('d')

    @classmethod
    def fromlist(cls, lst):
        cdef SparseArray res
        cdef Py_ssize_t i = 0
        cdef list index = []
        cdef list data = []
        [(index.append(k), data.append(v)) for k, v in enumerate(lst) if v != 0]
        res = cls.empty(len(lst), len(index))
        for j, d in zip(index, data):
            res.index.data.as_uints[i] = j
            res.data.data.as_doubles[i] = d
            i += 1
        return res
        
    def __len__(self):
        return self._len

    @classmethod
    def empty(cls, unsigned int len, unsigned int non_zero):
        cdef SparseArray res = cls()
        res._empty(len, non_zero)
        return res

    cdef void _empty(self, unsigned int len, unsigned int non_zero):
        self._len = len
        self.non_zero = non_zero
        array.resize(self.index, self.non_zero)
        array.resize(self.data, self.non_zero)

    cdef void fix_size(self, unsigned int new_size):
        array.resize(self.index, new_size)
        array.resize(self.data, new_size)
        self.non_zero = new_size

    cdef unsigned int union_size(self, SparseArray second):
        cdef Py_ssize_t i = 0, j = 0, c = 0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int b_non_zero = second.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef unsigned int *b = second.index.data.as_uints
        if self._len == a_non_zero and self._len == b_non_zero:
            return a_non_zero
        while (i < a_non_zero) and (j < b_non_zero):
            if a[i] == b[j]:
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
            c += 1
        if i == a_non_zero:
            return c + b_non_zero - j
        else:
            return c + a_non_zero - i
        return 0
    
    cdef SparseArray union_func(self, two_arguments func,
                                one_argument left,
                                one_argument right,
                                SparseArray second):
        cdef Py_ssize_t i = 0, j = 0, c = 0, k
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int b_non_zero = second.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef unsigned int *b = second.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef double *b_value = second.data.data.as_doubles
        cdef SparseArray res = self.empty(self._len, self.union_size(second))
        cdef double *output_data = res.data.data.as_doubles
        cdef unsigned int *output_index = res.index.data.as_uints
        cdef double res_value
        cdef unsigned int res_index
        if self._len == a_non_zero and self._len == b_non_zero:
            for k in range(a_non_zero):
                res_value = func(a_value[k], b_value[k])
                set_value(output_index, output_data, &c, a[k], res_value)
            if c < res.non_zero:
                res.fix_size(c)
            # res.non_zero = c
            return res
        while (i < a_non_zero) and (j < b_non_zero):
            if a[i] == b[j]:
                res_value = func(a_value[i], b_value[j])
                res_index = a[i]
                i += 1
                j += 1
            elif a[i] < b[j]:
                res_value = left(a_value[i])
                res_index = a[i]
                i += 1
            else:
                res_value = right(b_value[j])
                res_index = b[j]
                j += 1
            set_value(output_index, output_data, &c, res_index, res_value)
        if i == a_non_zero:
            for k in range(j, b_non_zero):
                set_value(output_index, output_data, &c, b[k],
                          right(b_value[k]))
        else:
            for k in range(i, a_non_zero):
                set_value(output_index, output_data, &c, a[k],
                          left(a_value[k]))
        if c < res.non_zero:
            res.fix_size(c)
        # res.non_zero = c
        return res

    cpdef SparseArray add(self, SparseArray second):
        return self.union_func(add_op, non_op, non_op, second)
    
    def __add__(self, second):
        return self.add(second)

    cpdef SparseArray sub(self, SparseArray second):
        return self.union_func(sub_op, non_op, minus_op, second)
    
    def __sub__(self, other):
        return self.sub(other)
        
    cpdef SparseArray min(self, SparseArray second):
        return self.union_func(math.fmin, zero_min_op, zero_min_op, second)

    cpdef SparseArray max(self, SparseArray second):
        return self.union_func(math.fmax, zero_max_op, zero_max_op, second)

    cpdef SparseArray div(self, SparseArray second):
        return self.union_func(div_op, div_left_op, div_right_op, second)

    def __div__(self, other):
        return self.div(other)

    def __truediv__(self, other):
        return self.div(other)

    cdef SparseArray one_argument_func(self, one_argument func, bint full):
        cdef SparseArray res
        cdef Py_ssize_t k, i=0, c=0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles        
        if full:
            res = self.empty(self._len, self._len)
        else:
            res = self.empty(self._len, self.non_zero)
        cdef double *output_data = res.data.data.as_doubles
        cdef unsigned int *output_index = res.index.data.as_uints
        cdef double res_value, zero_value = func(0)        
        if full:
            for k in range(self._len):
                if i < a_non_zero and a[i] == k:
                    res_value = func(a_value[i])
                    i += 1
                else:
                    res_value = zero_value
                set_value(output_index, output_data, &c, k, res_value)
        else:
            for k in range(self.non_zero):
                res_value = func(a_value[k])
                set_value(output_index, output_data, &c, a[k], res_value)
        if c < res.non_zero:
            res.fix_size(c)
        return res

    cpdef SparseArray sin(self):
        return self.one_argument_func(math.sin, 0)

    cpdef SparseArray cos(self):
        return self.one_argument_func(math.cos, 1)

    cpdef SparseArray tan(self):
        return self.one_argument_func(math.tan, 0)
    
    cpdef SparseArray asin(self):
        return self.one_argument_func(math.asin, 0)

    cpdef SparseArray acos(self):
        return self.one_argument_func(math.acos, 1)

    cpdef SparseArray atan(self):
        return self.one_argument_func(math.atan, 0)

    cpdef SparseArray sinh(self):
        return self.one_argument_func(math.sinh, 0)

    cpdef SparseArray cosh(self):
        return self.one_argument_func(math.cosh, 1)

    cpdef SparseArray tanh(self):
        return self.one_argument_func(math.tanh, 0)
    
    cpdef SparseArray asinh(self):
        return self.one_argument_func(math.asinh, 0)

    cpdef SparseArray atanh(self):
        return self.one_argument_func(math.atanh, 0)

    cpdef SparseArray exp(self):
        return self.one_argument_func(math.exp, 1)

    cpdef SparseArray expm1(self):
        return self.one_argument_func(math.expm1, 0)

    cpdef SparseArray log1p(self):
        return self.one_argument_func(math.log1p, 0)

    cpdef SparseArray sqrt(self):
        return self.one_argument_func(math.sqrt, 0)
    
    cpdef SparseArray fabs(self):
        return self.one_argument_func(math.fabs, 0)

    cpdef SparseArray ceil(self):
        return self.one_argument_func(math.ceil, 0)

    cpdef SparseArray floor(self):
        return self.one_argument_func(math.floor, 0)

    cpdef SparseArray trunc(self):
        return self.one_argument_func(math.trunc, 0)

    def __reduce__(self):
        return (rebuild, (self.data, self.index, self._len))
