# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:09:57 2017

@author: guoli
"""

def transpose(M):

    row = len(M)
    col = len(M[0])

#    print row
#    print col

    
    N = [['0']*row]*col

    for i in range(row):
        for j in range(col):
            N[j][i] = M[i][j]
      #      print M[i][j]
            print N[j][i]

    return N


A = [[1,2],[3,4],[5,6]]

print A

print transpose(A)
