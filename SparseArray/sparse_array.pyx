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


cimport cython
import types
from cpython cimport array
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM

cpdef rebuild(data, index, size):
    cdef SparseArray r = SparseArray()
    r._len = size
    r.data = data
    r.index = index
    r.non_zero = len(index)
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
        cdef unsigned int k
        cdef double v
        cdef Py_ssize_t i = 0
        try:            
            res = cls.empty(len(lst), len(lst) - lst.count(0))
        except AttributeError:
            k = 0
            for v in lst:
                if v == 0:
                    k += 1
            res = cls.empty(len(lst), len(lst) - k)
        for k, v in enumerate(lst):
            if v == 0:
                continue
            res.index.data.as_uints[i] = k
            res.data.data.as_doubles[i] = v
            i += 1
        return res

    @classmethod
    def index_data(cls, index_data, size):
        cdef SparseArray r = cls()
        r._len = size
        r.index = array.array('I', [x[0] for x in index_data])
        r.data = array.array('d', [x[1] for x in index_data])
        r.non_zero = len(index_data)
        return r

    @classmethod
    def constant(cls, value, index, size):
        cdef SparseArray r = cls()
        cdef double *data_value 
        cdef Py_ssize_t i, index_size = len(index)
        r._len = size
        r.non_zero = index_size
        r.index = array.array('I', index)
        r.data = array.clone(r.data, index_size, zero=False)
        data_value = r.data.data.as_doubles
        for i in range(index_size):
            data_value[i] = value
        return r
        
        
    @property
    def density(self):
        return self.non_zero / float(self._len)

    @property
    def used_memory(self):
        return self.data.itemsize * self.non_zero + self.index.itemsize * self.non_zero

    @property
    def maximum_memory(self):
        return self.data.itemsize * self._len + self.index.itemsize * self._len

    cdef double getitem(self, unsigned int k):
        cdef unsigned int start=0, end=self.non_zero, half
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        if k >= self._len:
            return 0
        while end - start > 1:
            half = start + (end - start) // 2
            if a[half] > k:
                end = half
            elif a[half] < k:
                start = half
            elif a[half] == k:
                return a_value[half]
            else:
                break
        if a[start] == k:
            return a_value[start]
        elif a[end] == k:
            return a_value[end]
        return 0

    def __getitem__(self, i):
        return self.getitem(i)
    
    def __len__(self):
        return self._len

    def size(self):
        return self._len

    def full_array(self):
        cdef array.array res = array.clone(self.data, self._len, zero=True)
        cdef Py_ssize_t i
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef double *output_data = res.data.as_doubles
        for i in range(a_non_zero):
            output_data[a[i]] = a_value[i]
        return res
 
    @classmethod
    def empty(cls, unsigned int len, unsigned int non_zero):
        cdef SparseArray res = cls()
        res._empty(len, non_zero)
        return res

    cdef void _empty(self, unsigned int len, unsigned int non_zero):
        self._len = len
        self.non_zero = non_zero
        self.index = array.clone(self.index, self.non_zero, zero=False)
        self.data = array.clone(self.data, self.non_zero, zero=False)

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
            return res
        while (i < a_non_zero) and (j < b_non_zero):
            res_index = a[i]            
            if res_index == b[j]:
                res_value = func(a_value[i], b_value[j])
                i += 1
                j += 1
            elif res_index < b[j]:
                res_value = left(a_value[i])
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
        return res

    cpdef SparseArray add(self, SparseArray second):
        return self.union_func(add_op, non_op, non_op, second)

    cpdef SparseArray add2(self, double second):
        cdef SparseArray res = self.empty(self._len, self._len)
        cdef Py_ssize_t k, i=0, c=0, j=0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles        
        cdef double *output_data = res.data.data.as_doubles
        cdef unsigned int *output_index = res.index.data.as_uints
        for k in range(self._len):
            if j < a_non_zero and a[j] == k:
                res_value = a_value[j] + second
                j += 1
            else:
                res_value = second
            set_value(output_index, output_data, &c, k, res_value)
        if c < res.non_zero:
            res.fix_size(c)
        return res
    
    def __add__(self, second):
        if isinstance(second, SparseArray):
            try:
                return self.add(second)
            except AttributeError:
                return second.add2(self)
        return self.add2(second)

    cpdef SparseArray sub(self, SparseArray second):
        return self.union_func(sub_op, non_op, minus_op, second)

    def __sub__(self, other):
        return self.sub(other)

    cpdef SparseArray min(self, SparseArray second):
        return self.union_func(math.fmin, zero_min_op, zero_min_op, second)

    cpdef SparseArray max(self, SparseArray second):
        return self.union_func(math.fmax, zero_max_op, zero_max_op, second)

    cpdef SparseArray div(self, SparseArray second):
        return self.full_func(div_op, div_left_op, div_right_op, second)
    # return self.union_func(div_op, div_left_op, div_right_op, second)


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

    cpdef SparseArray atan2(self, SparseArray second):
        return self.union_func(math.atan2, atan2_left_op,
                               atan2_right_op, second)
    
    cpdef SparseArray sinh(self):
        return self.one_argument_func(math.sinh, 0)

    cpdef SparseArray cosh(self):
        return self.one_argument_func(math.cosh, 1)

    cpdef SparseArray tanh(self):
        return self.one_argument_func(math.tanh, 0)
    
    cpdef SparseArray asinh(self):
        return self.one_argument_func(math.asinh, 0)

    cpdef SparseArray acosh(self):
        return self.one_argument_func(math.acosh, 1)

    cpdef SparseArray atanh(self):
        return self.one_argument_func(math.atanh, 0)

    cpdef SparseArray hypot(self, SparseArray second):
        return self.union_func(math.hypot, math.fabs,
                               math.fabs, second)

    cpdef SparseArray log(self):
        return self.one_argument_func(math.log, 1)

    cpdef SparseArray log2(self):
        return self.one_argument_func(math.log2, 1)

    cpdef SparseArray log10(self):
        return self.one_argument_func(math.log10, 1)

    cpdef SparseArray log1p(self):
        return self.one_argument_func(math.log1p, 0)

    cpdef SparseArray lgamma(self):
        return self.one_argument_func(math.lgamma, 1)
    
    cpdef SparseArray exp(self):
        return self.one_argument_func(math.exp, 1)

    cpdef SparseArray expm1(self):
        return self.one_argument_func(math.expm1, 0)

    cdef SparseArray full_func(self, two_arguments func,
                               one_argument left,
                               one_argument right,                               
                               SparseArray second):
        cdef SparseArray res = self.empty(self._len, self._len)
        cdef Py_ssize_t k, i=0, c=0, j=0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int b_non_zero = second.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef unsigned int *b = second.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef double *b_value = second.data.data.as_doubles
        cdef double *output_data = res.data.data.as_doubles
        cdef double zero_value = func(0, 0)
        cdef unsigned int *output_index = res.index.data.as_uints
        for k in range(self._len):
            if i < a_non_zero and j < b_non_zero:
                if a[i] == k:
                    if k == b[j]:
                        res_value = func(a_value[i], b_value[j])
                        j += 1
                    else:
                        res_value = left(a_value[i])
                    i += 1    
                elif b[j] == k:
                    res_value = right(b_value[j])
                    j += 1
                else:
                    res_value = zero_value
            elif i < a_non_zero:
                if a[i] == k:
                    res_value = left(a_value[i])
                    i += 1
                else:
                    res_value = zero_value
            elif j < b_non_zero:
                if b[j] == k:
                    res_value = right(b_value[j])
                    j += 1
                else:
                    res_value = zero_value
            else:
                res_value = zero_value
            set_value(output_index, output_data, &c, k, res_value)
        if c < res.non_zero:
            res.fix_size(c)
        return res

    cpdef SparseArray pow(self, SparseArray second):
        return self.full_func(math.pow, pow_left_op, pow_right_op, second)
    
    cpdef SparseArray sqrt(self):
        return self.one_argument_func(math.sqrt, 0)

    cpdef SparseArray sq(self):
        return self.one_argument_func(sq_op, 0)

    cpdef SparseArray sign(self):
        return self.one_argument_func(sign_op, 0)

    cpdef SparseArray boundaries(self):
        return self.one_argument_func(boundaries_op, 0)

    cpdef SparseArray copy(self):
        cdef SparseArray r = SparseArray()
        r._len = self._len
        r.index = array.copy(self.index)
        r.data = array.copy(self.data)
        r.non_zero = self.non_zero
        return r
    
    
    cpdef SparseArray fabs(self):
        return self.one_argument_func(math.fabs, 0)

    cpdef SparseArray ceil(self):
        return self.one_argument_func(math.ceil, 0)

    cpdef SparseArray floor(self):
        return self.one_argument_func(math.floor, 0)

    cpdef SparseArray trunc(self):
        return self.one_argument_func(math.trunc, 0)

    cpdef SparseArray erf(self):
        return self.one_argument_func(math.erf, 0)

    cpdef SparseArray erfc(self):
        return self.one_argument_func(math.erfc, 1)
    
    cpdef SparseArray finite(self, bint inplace=False):
        if not inplace:
            return self.one_argument_func(finite_op, 0)
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef Py_ssize_t i = 0, j = 0
        for i in range(a_non_zero):
            if math.isfinite(a_value[i]):
                if i > j:
                    a_value[j] = a_value[i]
                    a[j] = a[i]
                j += 1
        if j < a_non_zero:
            self.fix_size(j)
        return self
 
    cdef unsigned int intersection_size(self, SparseArray second):
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
                c += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return c

    cdef SparseArray intersection_func(self, two_arguments func,
                                       SparseArray second):
        cdef Py_ssize_t i = 0, j = 0, c = 0, k
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int b_non_zero = second.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef unsigned int *b = second.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef double *b_value = second.data.data.as_doubles
        cdef SparseArray res = self.empty(self._len, self.intersection_size(second))
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
            return res
        while (i < a_non_zero) and (j < b_non_zero):
            if a[i] == b[j]:
                res_value = func(a_value[i], b_value[j])
                res_index = a[i]
                set_value(output_index, output_data, &c, res_index, res_value)
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        if c < res.non_zero:
            res.fix_size(c)
        return res
    
    cpdef SparseArray mul(self, SparseArray second):
        return self.intersection_func(mul_op, second)

    cpdef SparseArray mul2(self, double second):
        cdef SparseArray res = self.empty(self._len, self.non_zero)
        cdef Py_ssize_t k, i=0, c=0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles        
        cdef double *output_data = res.data.data.as_doubles
        cdef unsigned int *output_index = res.index.data.as_uints
        for k in range(self.non_zero):
            res_value = a_value[k] * second
            set_value(output_index, output_data, &c, a[k], res_value)
        if c < res.non_zero:
            res.fix_size(c)
        return res
    
    def __mul__(self, second):
        if isinstance(second, SparseArray):
            try:
                return self.mul(second)
            except AttributeError:
                return second.mul2(self)
        return self.mul2(second)

    cpdef double dot(self, SparseArray second):
        cdef Py_ssize_t i = 0, j = 0, c = 0, k
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int b_non_zero = second.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef unsigned int *b = second.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles
        cdef double *b_value = second.data.data.as_doubles
        cdef double res_value = 0
        if self._len == a_non_zero and self._len == b_non_zero:
            for k in range(a_non_zero):
                res_value += a_value[k] * b_value[k]
            return res_value
        while (i < a_non_zero) and (j < b_non_zero):
            if a[i] == b[j]:
                res_value += a_value[i] * b_value[j]
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return res_value
         
    cpdef bint isfinite(self):
        cdef double *a_value = self.data.data.as_doubles
        cdef Py_ssize_t i = 0
        for i in range(self.non_zero):
            if not math.isfinite(a_value[i]):
                return False
        return True

    @staticmethod
    def argmax(list lst):
        cdef Py_ssize_t i, k, j
        cdef SparseArray next
        next = <SparseArray> PyList_GET_ITEM(lst, 0)
        cdef array.array best = next.full_array()
        cdef double *best_C = best.data.as_doubles, *comp, value
        cdef array.array best_index = array.clone(next.index, next._len, zero=True)
        cdef unsigned int *best_index_C = best_index.data.as_uints, *comp_index
        cdef unsigned int end, len=next._len
        for j in range(1, PyList_GET_SIZE(lst)):
            next = <SparseArray> PyList_GET_ITEM(lst, j)
            comp = next.data.data.as_doubles
            comp_index = next.index.data.as_uints
            end = next.non_zero
            k = 0
            for i in range(len):
                value = 0
                if k < end:
                    if comp_index[k] == i:
                        value = comp[k]
                        k += 1
                if value > best_C[i]:
                    best_C[i] = value
                    best_index_C[i] = j
        return SparseArray.fromlist(best_index)            

    @staticmethod
    def argmin(list lst):
        cdef Py_ssize_t i, k, j
        cdef SparseArray next
        next = <SparseArray> PyList_GET_ITEM(lst, 0)
        cdef array.array best = next.full_array()
        cdef double *best_C = best.data.as_doubles, *comp, value
        cdef array.array best_index = array.clone(next.index, next._len, zero=True)
        cdef unsigned int *best_index_C = best_index.data.as_uints, *comp_index
        cdef unsigned int end, len=next._len
        for j in range(1, PyList_GET_SIZE(lst)):
            next = <SparseArray> PyList_GET_ITEM(lst, j)
            comp = next.data.data.as_doubles
            comp_index = next.index.data.as_uints
            end = next.non_zero
            k = 0
            for i in range(len):
                value = 0
                if k < end:
                    if comp_index[k] == i:
                        value = comp[k]
                        k += 1
                if value < best_C[i]:
                    best_C[i] = value
                    best_index_C[i] = j
        return SparseArray.fromlist(best_index)            

    cpdef double sum(self):
        cdef double *a_value = self.data.data.as_doubles
        cdef double res = 0
        cdef Py_ssize_t i = 0
        for i in range(self.non_zero):
            res += a_value[i]
        return res

    @cython.cdivision(True)
    cpdef SparseArray unit_vector(self):
        cdef SparseArray res = self.empty(self._len, self.non_zero)
        cdef Py_ssize_t k, i=0, c=0
        cdef unsigned int a_non_zero = self.non_zero
        cdef unsigned int *a = self.index.data.as_uints
        cdef double *a_value = self.data.data.as_doubles        
        cdef double *output_data = res.data.data.as_doubles
        cdef unsigned int *output_index = res.index.data.as_uints
        cdef double res_value
        cdef double norm = math.sqrt(self.dot(self))
        for k in range(self.non_zero):
            res_value = a_value[k] / norm
            set_value(output_index, output_data, &c, a[k], res_value)
        if c < res.non_zero:
            res.fix_size(c)
        return res
    
    cpdef double cosine_distance(self, SparseArray second):
        cdef double p = self.dot(second)
        cdef double p1 = math.sqrt(self.dot(self))
        cdef double p2 = math.sqrt(second.dot(second))
        if p == 0 or p1 == 0 or p2 == 0:
            return 1.0
        p = p / (p1 * p2)
        return 1 - math.fabs(p)
    
    cpdef double pearson_coefficient(self, SparseArray second):
        cdef double xy = 0
        cdef double x = 0
        cdef double y = 0
        cdef double *x_value = self.data.data.as_doubles
        cdef double *y_value = second.data.data.as_doubles
        cdef unsigned int *x_index = self.index.data.as_uints
        cdef unsigned int *y_index = second.index.data.as_uints
        cdef double xm = self.sum() / self._len
        cdef double ym = second.sum() / second._len
        cdef Py_ssize_t i = 0, j = 0, k
        cdef Py_ssize_t cont = 0
        for k in range(self.non_zero):
            x+= sq_op(x_value[k]-xm)
        x+= (self._len-self.non_zero)*sq_op(xm)
        for k in range(second.non_zero):
            y+= sq_op(y_value[k]-ym)
        y+= (second._len-second.non_zero)*sq_op(ym)
        while (i<self.non_zero) and (j<second.non_zero):
            if x_index[i] == y_index[j]:
                xy+= (x_value[i]-xm)*(y_value[j]-ym)
                i += 1
                j += 1
            elif x_index[i] < y_index[j]:
                xy+= (x_value[i]-xm)*(-ym)
                i += 1
            elif x_index[i] > y_index[j]:
                xy+= (-xm)*(y_value[j]-ym)
                j += 1
            cont += 1
        if i==self.non_zero:
            for k in range(j,second.non_zero):
                xy+= (-xm)*(y_value[k]-ym)
                cont += 1
        else:
            for k in range(i,self.non_zero):
                xy+= (x_value[k]-xm)*(-ym)
                cont += 1
        xy+= (second._len-cont)*(-xm)*(-ym)
        return xy/math.sqrt(x*y)

    cpdef double SSE(self, SparseArray second):
        return self.sub(second).sq().sum()

    cpdef double SAE(self, SparseArray second):
        return self.sub(second).fabs().sum()
    
    @staticmethod
    def cumsum(list lst):
        cdef SparseArray r
        a = (lst[0]).add(lst[1])
        for r in lst[2:]:
            a = a.add(r)
        return a

    @staticmethod
    def cummul(list lst):
        cdef SparseArray r
        a = (lst[0]).mul(lst[1])
        for r in lst[2:]:
            a = a.mul(r)
        return a

    @staticmethod
    def cummin(list lst):
        cdef SparseArray r
        a = (lst[0]).min(lst[1])
        for r in lst[2:]:
            a = a.min(r)
        return a

    @staticmethod
    def cummax(list lst):
        cdef SparseArray r
        a = (lst[0]).max(lst[1])
        for r in lst[2:]:
            a = a.max(r)
        return a
    
    def __reduce__(self):
        return (rebuild, (self.data, self.index, self._len))
