# -*- coding: utf-8 -*-

from vector import Vector


v1 = Vector([8.462,7.893,-8.187])
v2 = Vector([6.984,-5.975,4.778])

v3 = Vector([-8.987,-9.838,5.031])
v4 = Vector([-4.268,-1.861,-8.866])

v5 = Vector([1.5,9.547,3.691])
v6 = Vector([-6.007,0.124,5.772])


print "***********************"

print v1.vector_multiple(v2)
print v3.area_parallelogram(v4)
print v5.area_trangle(v6)

