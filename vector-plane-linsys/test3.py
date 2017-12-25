# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:17:18 2017

@author: guoli
"""

def matxMultiply(A,B):

    C = [[0 for x in range(len(B))] for y in range(len(A[0]))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C


A = [[-2, -10, 3, 1, 8],
     [-4, 8, 5, 7, 6],
     [-5, 8, -5, -7, -10],
     [2, 2, 2, 5, 2],
     [9, -2, -3, -10, -8],
     [6, 5, -4, 0, -3],
     [-8, -2, 3, 5, -7],
     [-5, 8, -1, -6, 7],
     [-2, -5, 7, 4, -8],
     [6, -1, -1, 8, 9],
     [7, -2, 3, -2, 2],
     [8, -9, -2, -7, 6],
     [-3, 0, 0, 9, -2],
     [2, 6, 8, 7, -9],
     [-2, 4, 4, -10, -8],
     [-10, -6, 8, 5, -5],
     [4, 8, -4, -7, 4]]

B = [[-3, 2, 2, 0, -5, -2, -1, 2, -4, -1, 2, -5, 3, -3, 3, 2],
     [-2, -1, 3, -5, -2, 1, -4, 3, -2, -4, -4, 2, 0, -4, 1, 4],
     [-1, 3, 1, -1, 0, -2, 0, -1, -4, -2, -3, -4, 1, 0, 2, 2],
     [1, 1, -3, 2, 0, -5, -2, -5, -2, 0, 4, 4, 3, 2, -3, -2],
     [-5, -4, 2, -2, 1, 1, -1, -2, -5, -5, -3, 3, -1, -4, 0, 2]]


print matxMultiply(A,B)