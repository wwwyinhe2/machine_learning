# -*- coding: utf-8 -*-

print "\033[1;33;44mTest triangular form below:\033[0m"

from plane import Plane
from vector import Vector
from linsys import LinearSystem
from decimal import Decimal


p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])

t = s.compute_triangular_form()
if not (t[0] == p1 and
        t[1] == p2):
    print 'test case 1 failed'
else:
    print "test case 1 pass"
"""    
print "******1**********"
print s
print "*****************"
print t
"""

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
t = s.compute_triangular_form()
if not (t[0] == p1 and
        t[1] == Plane(constant_term='1')):
    print 'test case 2 failed'
else:
    print "test case 2 pass"
"""
print "******2**********"
print s
print "*****************"
print t
"""
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
t = s.compute_triangular_form()
if not (t[0] == p1 and
        t[1] == p2 and
        t[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        t[3] == Plane()):
    print 'test case 3 failed'
else:
    print "test case 3 pass"
"""
print "******3**********"
print s
print "*****************"
print t
"""
p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
t = s.compute_triangular_form()
if not (t[0] == Plane(normal_vector=Vector(['1','-1','1']), constant_term='2') and
        t[1] == Plane(normal_vector=Vector(['0','1','1']), constant_term='1') and
        t[2] == Plane(normal_vector=Vector(['0','0','-9']), constant_term='-2')):
    print 'test case 4 failed'
else:
    print "test case 4 pass"
"""
print "******4**********"
print s
print "*****************"
print t
"""

print "\033[1;33;44m Test RREF below: \033[0m"

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='-1') and
        r[1] == p2):
    print 'test case 1 failed'
else:
    print "test case 1 pass"
  

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
r = s.compute_rref()
if not (r[0] == p1 and
        r[1] == Plane(constant_term='1')):
    print 'test case 2 failed'
else:
    print "test case 2 pass"

    
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='0') and
        r[1] == p2 and
        r[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        r[3] == Plane()):
    print 'test case 3 failed'
else:
    print "test case 3 pass"


p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term=Decimal('23')/Decimal('9')) and
        r[1] == Plane(normal_vector=Vector(['0','1','0']), constant_term=Decimal('7')/Decimal('9')) and
        r[2] == Plane(normal_vector=Vector(['0','0','1']), constant_term=Decimal('2')/Decimal('9'))):
    print 'test case 4 failed'
else:
    print "test case 4 pass"

print "\033[1;33;44m Solve linsys: \033[0m"

p1 = Plane(normal_vector=Vector(['5.862','1.178','-10.366']),constant_term='-8.15')
p2 = Plane(normal_vector=Vector(['-2.931','-0.589','5.183']),constant_term='-4.075')

s1 = LinearSystem([p1,p2])
print s1.compute_solution()

#print s1.compute_rref()
print "******************"
p3 = Plane(normal_vector=Vector(['8.631','5.112','-1.816']),constant_term='-5.113')
p4 = Plane(normal_vector=Vector(['4.315','11.132','-5.27']),constant_term='-6.775')
p5 = Plane(normal_vector=Vector(['-2.158','3.01','-1.727']),constant_term='-0.831')

s2 = LinearSystem([p3,p4,p5])
#print s2.compute_rref()
print s2.compute_solution()
print "******************"

p6 = Plane(normal_vector=Vector(['5.262','2.739','-9.878']),constant_term='-3.441')
p7 = Plane(normal_vector=Vector(['5.111','6.358','7.638']),constant_term='-2.152')
p8 = Plane(normal_vector=Vector(['2.016','-9.924','-1.367']),constant_term='-9.278')
p9 = Plane(normal_vector=Vector(['2.167','-13.543','-18.883']),constant_term='-10.567')

s3 = LinearSystem([p6,p7,p8,p9])
#print s3.compute_rref()
print s3.compute_solution()











