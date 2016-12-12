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


def random_lst(size=100, p=0.5):
    lst = []
    for i in range(size):
        if random() < p:
            lst.append(random())
        else:
            lst.append(0)
    return lst


def test_fromlist():
    lst = random_lst()
    array = SparseArray.fromlist(lst)
    [assert_almost_equals(a, b) for a, b in zip([x for x in lst if x != 0],
                                                array.data)]


def test_empty():
    array = SparseArray.empty(100, 10)
    assert len(array) == 100
    assert len(array.index) == 10


def test_sum():
    for p in [0.5, 1]:
        a = random_lst(p=p)
        b = random_lst(p=p)
        a[10] = 12.433
        b[10] = -12.433
        c = SparseArray.fromlist(a) + SparseArray.fromlist(b)
        res = [x + y for x, y in zip(a, b)]
        index = [k for k, v in enumerate(res) if v != 0]
        res = [x for x in res if x != 0]
        assert c.non_zero == len(res)
        assert len(c.data) == c.non_zero
        [assert_almost_equals(v, w) for v, w in zip(index,
                                                    c.index)]
        print(c.non_zero, len(c.data), len([x for x in res if x != 0]))
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
    from math import sinh, cosh, tanh, asinh, atanh
    from math import exp, expm1, log1p, sqrt
    from math import fabs, ceil, floor, trunc
    for f in [sin, cos, tan, asin, acos, atan,
              sinh, cosh, tanh, asinh, atanh,
              exp, expm1, log1p, sqrt,
              fabs, ceil, floor, trunc]:
        for p in [0.5, 1]:
            a = random_lst(p=p)
            b = SparseArray.fromlist(a)
            c = getattr(b, f.__name__)()
            res = [f(x) for x in a]
            index = [k for k, v in enumerate(res) if v != 0]
            res = [x for x in res if x != 0]
            print(f, p, c.non_zero, len(res))
            assert c.non_zero == len(res)
            [assert_almost_equals(v, w) for v, w in zip(index,
                                                        c.index)]
            [assert_almost_equals(v, w) for v, w in zip(res,
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
    from math import isfinite
    a = SparseArray.fromlist(random_lst(p=0.5))
    b = SparseArray.fromlist(random_lst(p=0.5))
    c = a / b
    res = [i for i in c.data if isfinite(i)]
    d = c.finite()
    [assert_almost_equals(v, w) for v, w in zip([x for x in res if x != 0],
                                                d.data)]
    
