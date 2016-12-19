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


from SparseArray import SparseArray
from nose.tools import assert_almost_equals
from random import random
try:
    from math import isfinite
except ImportError:
    import math

    def isfinite(x):
        if math.isinf(x) or math.isnan(x):
            return False
        return True



def random_lst(size=100, p=0.5):
    lst = []
    for i in range(size):
        if random() < p:
            c = 1 if random() < 0.5 else -1
            lst.append(random() * c)
        else:
            lst.append(0)
    return lst


def test_fromlist():
    lst = random_lst()
    array = SparseArray.fromlist(lst)
    [assert_almost_equals(a, b) for a, b in zip([x for x in lst if x != 0],
                                                array.data)]


def test_len_size():
    lst = random_lst()
    array = SparseArray.fromlist(lst)
    assert len(array) == array.size()


def test_index_data():
    lst = random_lst()
    array = SparseArray.index_data([(k, v) for k, v in enumerate(lst) if v != 0],
                                   len(lst))
    [assert_almost_equals(a, b) for a, b in zip([x for x in lst if x != 0],
                                                array.data)]


def test_empty():
    array = SparseArray.empty(100, 10)
    assert len(array) == 100
    assert len(array.index) == 10


def test_two_args():
    from math import atan2, hypot

    def add(a, b):
        return a + b

    for f, name in zip([add, atan2, hypot],
                       ['add', 'atan2', 'hypot']):
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = random_lst(p=p)
            a[10] = 12.433
            b[10] = -12.433
            c = getattr(SparseArray.fromlist(a),
                        name)(SparseArray.fromlist(b))
            res = [f(x, y) for x, y in zip(a, b)]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            print(c.non_zero, len(res), f)
            assert c.non_zero == len(res)
            assert len(c.data) == c.non_zero
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                        c.data)]


def test_sub():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        a[10] = 12.433
        b[10] = 12.433
        c = SparseArray.fromlist(a) - SparseArray.fromlist(b)
        res = [x - y for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert c.non_zero == len(res)
        assert len(c.data) == c.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    c.data)]


def test_min():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        a[10] = 12.433
        b[10] = 0
        c = SparseArray.fromlist(a).min(SparseArray.fromlist(b))
        res = [min(x, y) for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert c.non_zero == len(res)
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    c.data)]


def test_max():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        a[10] = 12.433
        b[10] = 0
        c = SparseArray.fromlist(a).max(SparseArray.fromlist(b))
        res = [max(x, y) for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert c.non_zero == len(res)
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    c.data)]


def test_div():
    def div(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            if a == b:
                return float('nan')
            return float('inf')
        
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        c = SparseArray.fromlist(a) / SparseArray.fromlist(b)
        res = [div(x, y) for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res)
                 if v != 0 and (a[k] != 0 or b[k] != 0)]
        res = [res[x] for x in index]
        assert c.non_zero == len(res)
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    c.data)]


def test_one():
    from math import sin, cos, tan, asin, acos, atan
    from math import sinh, cosh, tanh, asinh, acosh, atanh
    from math import exp, expm1, log, log10, log1p, sqrt, lgamma
    from math import fabs, ceil, floor, trunc
    try:
        from math import log2
    except ImportError:
        def log2(x):
            return log(x) / log(2)

    def wrapper(f, v):
        try:
            return f(v)
        except ValueError:
            if f == sqrt:
                return float('nan')
            if v >= 0:
                return float('inf')
            else:
                return -float('inf')

    def compare(a, b):
        if isfinite(a) and isfinite(b):
            return assert_almost_equals(a, b)
        return str(a) == str(b)

    for f in [sin, cos, tan, asin, acos, atan,
              sinh, cosh, tanh, asinh, acosh, atanh,
              exp, expm1, log, log2, log10, log1p, sqrt,
              lgamma, 
              fabs, ceil, floor, trunc]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = SparseArray.fromlist(a)
            c = getattr(b, f.__name__)()
            res = [wrapper(f, x) for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            print(f, p, c.non_zero, len(res))
            assert c.non_zero == len(res)
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            [compare(v, w) for v, w in zip(res,
                                           c.data)]


def test_mul():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        c = SparseArray.fromlist(a) * SparseArray.fromlist(b)
        res = [x * y for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert c.non_zero == len(res)
        assert len(c.data) == c.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    c.data)]


def test_isfinite():
    a = SparseArray.fromlist(random_lst(p=0.5))
    b = SparseArray.fromlist(random_lst(p=0.5))
    c = a / b
    assert a.isfinite()
    assert b.isfinite()
    assert not c.isfinite()


def test_sum2():
    a = random_lst(p=0.5)
    res = sum([x for x in a])
    c = SparseArray.fromlist(a).sum()
    assert res == c


def test_sq():
    a = random_lst(p=0.5)
    res = [x**2 for x in a]
    index = [k for k, v in enumerate(res) if v != 0]
    res = [x for x in res if x != 0]
    c = SparseArray.fromlist(a).sq()
    assert c.non_zero == len(res)
    assert len(c.data) == c.non_zero
    [assert_almost_equals(v, w) for v, w in zip(index,
                                                c.index)]
    print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                c.data)]


def test_sign():
    def sign(a):
        if a > 0:
            return 1
        elif a < 0:
            return -1
        return 0
    
    a = random_lst(p=0.5)
    res = [sign(x) for x in a]
    index = [k for k, v in enumerate(res) if v != 0]
    res = [x for x in res if x != 0]
    c = SparseArray.fromlist(a).sign()
    assert c.non_zero == len(res)
    assert len(c.data) == c.non_zero
    [assert_almost_equals(v, w) for v, w in zip(index,
                                                c.index)]
    print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                c.data)]


def test_unit_vector():
    from math import sqrt
    a = random_lst(p=0.5)
    norm = sqrt(sum([x**2 for x in a]))
    res = [x / norm for x in a]
    index = [k for k, v in enumerate(res) if v != 0]
    res = [x for x in res if x != 0]
    c = SparseArray.fromlist(a).unit_vector()
    assert c.non_zero == len(res)
    assert len(c.data) == c.non_zero
    [assert_almost_equals(v, w) for v, w in zip(index,
                                                c.index)]
    print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                c.data)]


def test_finite():
    a = SparseArray.fromlist(random_lst(p=0.5))
    b = SparseArray.fromlist(random_lst(p=0.5))
    c = a / b
    res = [i for i in c.data if isfinite(i)]
    d = c.finite()
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                d.data)]
    

def test_cumsum():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        c = random_lst(p=p)
        
        d = SparseArray.cumsum([SparseArray.fromlist(a),
                                SparseArray.fromlist(b),
                                SparseArray.fromlist(c)])
        res = [x + y + z for x, y, z in zip(a, b, c)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert d.non_zero == len(res)
        assert len(d.data) == d.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    d.index)]
        print(d.non_zero, len(d.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    d.data)]


def test_cummin():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        c = random_lst(p=p)

        d = SparseArray.cummin([SparseArray.fromlist(a),
                                SparseArray.fromlist(b),
                                SparseArray.fromlist(c)])
        res = [min([x, y, z]) for x, y, z in zip(a, b, c)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert d.non_zero == len(res)
        assert len(d.data) == d.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    d.index)]
        print(d.non_zero, len(d.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    d.data)]
        

def test_cummax():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        c = random_lst(p=p)

        d = SparseArray.cummax([SparseArray.fromlist(a),
                                SparseArray.fromlist(b),
                                SparseArray.fromlist(c)])
        res = [max([x, y, z]) for x, y, z in zip(a, b, c)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert d.non_zero == len(res)
        assert len(d.data) == d.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    d.index)]
        print(d.non_zero, len(d.data), len([x for x in res if x != 0]))
        [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                    d.data)]


def test_mul_const():
    for k in [32.4, 0]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = k
            c = SparseArray.fromlist(a) * b
            res = [x * b for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            assert c.non_zero == len(res)
            assert len(c.data) == c.non_zero
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
            [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                        c.data)]


def test_mul_const2():
    for k in [32.4, 0]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = k
            c = b * SparseArray.fromlist(a)
            res = [x * b for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            assert c.non_zero == len(res)
            assert len(c.data) == c.non_zero
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
            [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                        c.data)]


def test_sum_const():
    for k in [32.4, 0]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = k
            c = SparseArray.fromlist(a) + b
            res = [x + b for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            assert c.non_zero == len(res)
            assert len(c.data) == c.non_zero
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
            [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                        c.data)]


def test_sum_const2():
    for k in [32.4, 0]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = k
            c = b + SparseArray.fromlist(a)
            res = [x + b for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            assert c.non_zero == len(res)
            assert len(c.data) == c.non_zero
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
            [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                        c.data)]


def test_pickle():
    import pickle
    import tempfile
    suno = SparseArray.fromlist(random_lst())
    with tempfile.TemporaryFile('w+b') as io:
        pickle.dump(suno, io)
        io.seek(0)
        s = pickle.load(io)
        assert s.SSE(suno) == 0
            

def test_SSE():
    a = random_lst(p=0.5)
    b = random_lst(p=0.5)
    res = sum([(x - y)**2 for x, y in zip(a, b)])
    assert res == SparseArray.fromlist(a).SSE(SparseArray.fromlist(b))


def test_SAE():
    from math import fabs
    a = random_lst(p=0.5)
    b = random_lst(p=0.5)
    res = sum([fabs(x - y) for x, y in zip(a, b)])
    assert res == SparseArray.fromlist(a).SAE(SparseArray.fromlist(b))
    

def test_density():
    a = random_lst(p=0.5)
    density = (len(a) - a.count(0)) / float(len(a))
    b = SparseArray.fromlist(a)
    assert b.density == density


def test_full_array():
    a = random_lst(p=0.5)
    b = SparseArray.fromlist(a)
    [assert_almost_equals(v, w) for v, w in zip(a,
                                                b.full_array())]


def test_used_maximum_memory():
    a = random_lst(p=0.5)
    b = SparseArray.fromlist(a)
    assert b.used_memory
    assert b.maximum_memory


def test_boundaries():
    def boundaries(a):
        if a > 1:
            return 1
        elif a < -1:
            return -1
        return a
    
    a = random_lst(p=0.5)
    a[10] = 100
    a[11] = -32.3
    res = [boundaries(x) for x in a]
    index = [k for k, v in enumerate(res) if v != 0]
    res = [x for x in res if x != 0]
    c = SparseArray.fromlist(a).boundaries()
    assert c.non_zero == len(res)
    assert len(c.data) == c.non_zero
    [assert_almost_equals(v, w) for v, w in zip(index,
                                                c.index)]
    print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                c.data)]
    

def test_copy():
    a = random_lst(p=0.5)
    b = SparseArray.fromlist(a)
    c = b.copy()
    assert b.SSE(c) == 0


def test_dot():
    for p in [0.5, 1]:
        a = SparseArray.fromlist(random_lst(p=p))
        b = SparseArray.fromlist(random_lst(p=p))
        assert (a * b).sum() == a.dot(b)
    
