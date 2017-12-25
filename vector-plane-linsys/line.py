# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:44:40 2017

@author: guoli
"""
from decimal import Decimal, getcontext

from vector import Vector

getcontext().prec = 30


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Line.first_nonzero_index(n)
            initial_coefficient = n[initial_index]

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector

        try:
            initial_index = Line.first_nonzero_index(n)
            terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)
        
#%%
    def parallel(self,other):
        """
        check two lines whether parallel.
        input: line
        output:boolean
        """
        return self.normal_vector.parallel(other.normal_vector)
    
#%%
    def __eq__(self,other):
        """
        check tow lines wheter equal(the same line)
        input:line
        output:boolean
        """
        if self.normal_vector.is_zero():
            if other.normal_vector.is_zero():
                return True
            else:
                return False
        elif other.normal_vector.is_zero():
            return False
        else:
            n1 = self.basepoint
            n2 = other.basepoint
            n = n1.minus(n2)
            return self.normal_vector.orthogonality(n)

#%%   
    def intersection_with(self,other):
        """
        if have intersection, calculate the intersection point
        input: line
        output:vector
        
        """
        try:
           A,B = self.normal_vector.coordinates
           C,D = other.normal_vector.coordinates
           k1 = self.constant_term
           k2 = other.constant_term
           x = D*k1 - B*k2
           y = -C*k1 + A*k2
           num = Decimal('1')/(A*D-B*C)
           
           return Vector([x,y]).times_scalar(num)
       
        except ZeroDivisionError:
           if self == other:
               return self
           else:
               return None
    



#%%

class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps
