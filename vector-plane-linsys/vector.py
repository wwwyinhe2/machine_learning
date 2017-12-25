# -*- coding: utf-8 -*-

from math import acos, pi
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):
    
    ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = "ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG"
    
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)
            self.idx = 0

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __iter__(self):
        """
        changed here to turn to Iterator
        old: return self
        new: iter(self.coordinates)
        
        """
        return iter(self.coordinates)

    def next(self):
       self.idx += 1
       try:
           # delete this Decimal here,because I add one above.
           return self.coordinates[self.idx-1]
       except IndexError:
           self.idx = 0
           raise StopIteration  # Done iterating.

    def __getitem__(self,index):
        return Decimal(self.coordinates[index])

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

#%%
    def __eq__(self, v):
        """
        两个向量是否相等
        compare two vectors.
        input: vector
        output:boolean
        """
        return self.coordinates == v.coordinates
    
#%%    
    def plus(self,v):
        """
        两个向量的和
        add two vectors
        input:vector
        output:vector(Decimal)       
        """
        new_coordinates = []
        for x, y in zip(self.coordinates,v.coordinates):
            new_coordinates.append(x+y)
        return Vector(new_coordinates)
    
#%%     
    def minus(self,v):
        """
        两个向量的减
        minus two vectors
        input:vectors
        output:vector(Decimal)       
        """
        new_coordinates = []
        for x, y in zip(self.coordinates,v.coordinates):
            new_coordinates.append(x-y)
        return Vector(new_coordinates)
    
#%%
    def times_scalar(self,num):
        """
        向量与系数的乘
        multiple coefficient to a vector
        input: int or float
        output:vector(Decimal)
        """
        new_coordinates = [Decimal(num) * x  for x in self.coordinates]
        return Vector(new_coordinates)
    
#%%
    # magnitude a vector(向量的长度)
    def magnitude(self):
        """
        向量的长度
        magnitude a vector
        input: vector
        ouput: Decimal
        """
        coordinates_squard = [Decimal(x)**2 for x in self.coordinates]
        return (sum(coordinates_squard)).sqrt()

#%%
    
    def normalized(self):
        """
        向量标准化
        normalize a vector
        input: vector
        ouput:vector
        
        """
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal('1.0')/magnitude)
        except ZeroDivisionError:
            raise Exception("Cannot normalize the zero vector")
            
            
#%%
    
    def dot(self,v):
        """
        两个向量的点积
        input:vector
        output:Decimal
        """
        return sum([x*y for x,y in zip(self.coordinates,v.coordinates)])
    
#%%
   
    def angle(self,v,in_degrees = False):
        """
        两个向量的角,有弧度和角度两种表示
        the angle of two vector
        input :vector
        ouput:Decimal
        
        """
        try:
            v1 = self.normalized()
            v2 = v.normalized()
            angle_in_radians = acos(v1.dot(v2))

        #    print v1.dot_product(v2)

            if in_degrees:
                degree_per_radian = 180./pi
                return degree_per_radian * angle_in_radians
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception("Cannot compute an angle with the zero vector")
            else:
                raise e
                
                
#%%
    
    def parallel(self,v):
        """
        检查两个向量是否平行
        check two vectors whether parallel
        input:vector
        ouput:boolean
        
        """
        if (self.is_zero() or v.is_zero() or self.angle(v) == 0 or self.angle(v) == pi):
            return True
        else:
            return False
        
#%%
    def is_zero (self,tolerance = 1e-10):
        """
        check whether is zero
        """
        return self.magnitude() < tolerance
    
#%%
   
    def orthogonality(self,v,tolerance = 1e-10):
        """
        检查两个向量是否正交
        check two vectors whether orthogonality
        input:vector
        ouput:boolean
        
        """
        return abs(self.dot(v)) < tolerance
    
#%%
    
    def projection(self,v):
        """
        计算一个向量投影到另外一个向量的向量投影
        calculate the projection  of a vector to other vector
        input: vector
        ouput:Decimal
        """
        try:
            u = v.normalized()
            dot = self.dot(u)
            # 通过计算投影长度和投影方向向量标准化的乘得到
            return u.times_scalar(dot)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e
                
#%%
    # calculate the vertical_vector of a vector to other vector.
    def vector_vertical(self,v):
        """
        计算一个向量相对另一个向量的垂直向量
        calculate the vertical_vector of a vector to other vector.
        input:vector
        ouput:vector
        """
        try:
            p = self.projection(v)
            return self.minus(p)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e
                
#%%
   
    def vector_multiple(self,v):
        """
        计算两个向量的x乘
        calculate the multiple of two vectors
        input:vector
        ouput:vector
        """       
        try:
            x1, y1, z1 = self.coordinates
            x2, y2, z2 = v.coordinates
            new_coordinates = [y1*z2 - y2*z1,
                               -(x1*z2-x2*z1),
                               x1*y2 - x2*y1]
            return Vector(new_coordinates)
        
        except ValueError as e:
            msg = str(e)
            if msg == "need more than 2 values to unpack":
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                v_embedded_in_R3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3.vector_multiple(v_embedded_in_R3)
            elif(msg == "too many values to unpack" or
                 msg == "need more than 1 value to unpack"):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
        else:
            raise e
    
#%%
   
    def area_parallelogram(self,v):
        """
        计算两个向量组成的平行四边形面积
        calculate the area of the parallelogram
        input:vector
        ouput:Decimal
        """
        v2 = self.vector_vertical(v)
        length = v.magnitude()
        height = v2.magnitude()
        return length * height
    
#%%
    
    def area_trangle(self,v):
        """
        计算两个向量形成的三角形的面积
        calculate the area of the triangle
        input:vector
        output:decimal
        """
        
        return Decimal('0.5')*(self.area_parallelogram(v))

