from cmu_112_graphics import *
import numpy as np
from numpy import random,sin,cos,pi
import copy
#Helper Functions###################
def almostEqual(a,b):
    eps = 10**-12
    return (abs(a-b)<=eps)
def removeRowAndCol(A, row, col):
    result = []
    for r in range (len(A)):
        if r != row:#remove row
            a = copy.copy(A[r])
            a.pop(col)#remove column
            result.append(a) 
    return result
def print2dList(a):
    if type(a).__name__ ==  "matrix":
        a = a.m
    if (a == []): print([]); return
    rows, cols = len(a), len(a[0])
    colWidths = [0] * cols
    for col in range(cols):
        colWidths[col] = max([len(str(a[row][col])) for row in range(rows)])
    print('[')
    for row in range(rows):
        print(' [ ', end='')
        for col in range(cols):
            if (col > 0): print(', ', end='')
            print(str(a[row][col]).ljust(colWidths[col]), end='')
        print(' ]')
    print(']')
def zeros(m,n):
    result = []
    for i in range(m):
        result.append([0]*n)
    return result
def dotProduct (v1,v2):#gives dot product of vectors v1 v2
    if len(v1) != len(v2):
        return None
    sum = 0
    for i in range (len(v1)):
        sum  += v1[i]*v2[i]
    return sum
def randVector(bounds,length):#returns column vector of random elements inside bounds
    result = []
    lower = min(bounds)
    upper = max(bounds)

    for i in range(length):
        r = random.uniform(lower,upper)
        result.append([r])
    return result
############################################

class matrix(object):
    def __init__(self, m):
        self.m = m
        self.shape = (len(m),len(m[0]))

    def __eq__(self, other):
        return (self.m == other.m)
    def __repr__(self):
        return f"{self.m}"
    def add(self,other):
        if self.shape == other.shape:
            a = copy.deepcopy(self.m)
            b = copy.deepcopy(other.m)
            m,n = self.shape
            mNewList = zeros(m,n)
            rows,cols = self.shape
            for row in range(rows):
                for col in range(cols):
                    mNewList[row][col] = a[row][col] + b[row][col]
            mNew = matrix(mNewList)
            return mNew
    def scalarMultiply(self,c):
        m,n = self.shape
        mNewList = zeros(m,n)
        for i in range(m):
            for j in range(n):
                mNewList[i][j] = round(c*self.m[i][j],3)
        mNew = matrix(mNewList)
        return mNew
    def matrixMultiply(self,other):
    # C[i,k] = A[i,j]B[j,k]
    # A(m x n), B(o x l) , C (m x l) iff n=o
        m1 = copy.deepcopy(self.m)
        m2 = copy.deepcopy(other.m)
        m,n = self.shape# m1 rows
        o,l = other.shape# m1 cols
        if n != o:
            return None
        rows = m
        cols = l
        mNewList = zeros(m,l)
    #calculating elements
        for i in range(rows):
            for j in range(cols):
                rowVector = m1[i]
                colVector = [col[j] for col in m2] 
                mNewList[i][j] = dotProduct(rowVector,colVector)
        self.m = m1
        other.m = m2
        mNew = matrix(mNewList)
        return mNew
    def editCell(self,elem,row,col):
        self.m[row][col] = elem
    def addRow(self,newRow,pos):
        l = len(self.m)
        if pos <=0:
            self.m.insert(0,newRow)
        if pos > l:
            self.m.insert(l-1,newRow)
        self.shape = (self.shape[0]+1,self.shape[1])
    def removeRow(self,pos):
        self.m.pop(pos)
    def addCol(self,newCol,pos):
        l = len(self.m)
        for i in range(l):
            row = self.m[i] 
            row.insert(pos,newCol[i])
        self.shape = (self.shape[0],self.shape[1]+1)
    def isSquare(self):
        return (self.shape[0]==self.shape[1])
    def isSymmetric(self):
        if self.isSquare():
            m,n = self.shape
            a = self.m
            for i in range(m):
                for j in range(n):
                    if a[i][j] != a[j][i]:
                        return False
            return True
        else:
            return False 

    def det(self):#calculates determinate recursively
        if self.isSquare:
            a = self.m
            def innerDet(a):
                s = len(a[0])
                total = 0
                if s==2:    
                    a11 = a[0][0]
                    a12 = a[0][1]
                    a21 = a[1][0]
                    a22 = a[1][1]
                    return a11*a22-a12*a21
                elif s>2:
                    for i in range(s):
                        subMatrix = removeRowAndCol(a,0,i)
                        total += a[0][i]*innerDet(subMatrix)*(-1)**i
                    return total
            return innerDet(a)
        else:
            return None
    def trace(self):#calculates trace
        if self.isSquare():
            total = 0
            s = len(self.m)
            for i in range(s):
                total += self.m[i][i]
            return total
        else:
            return None
    def norm(self):#finds norm of column vector
        m,n = self.shape
        a = self.m
        if (n == 1 or m ==1):
            total = 0
            k = max(n,m)
            for i in range(k):
                total += a[i][0]**2
            return total**(.5)
        else:
            return None
    def transpose(self):#calculates transpose
        m,n = self.shape
        resultList = []
        for i in range(n):
            resultList.append([0]*m)
            for j in range(m):
                resultList[i][j] = self.m[j][i]
        result = matrix(resultList)
        return result

    def adjugate(self):#calculates adjugate of matrix
        b = copy.deepcopy(self.m)
        m,n = self.shape
        c = zeros(m,n)
        for i in range(m):
            for j in range(n):
                Mji = removeRowAndCol(b,j,i)
                d = matrix(Mji).det()
                if almostEqual(d,0):
                    d = 0
                c[i][j] = d*(-1)**(i+j) 
        C = matrix(c)
        return C

    def inverse(self):#calculates inverse of matrix
        if self.isSquare():
            d = self.det()
            adjugate = self.adjugate()
            if almostEqual(d,0) is False:
                adjugate.scalarMultiply(1/d)
                return adjugate
        else:
            return None
    def power(self,n):#multiplies matrix by itself n times
        if self.isSquare:
            a = self
            b = copy.deepcopy(a)
            if n <= 1:
                return a
            for i in range(1,n):
                #a = self.matrixMultiply(self)
               a =  a.matrixMultiply(b)
            return a
        else:
            return None
    def isEigenvector(self,v):#v is some column vector
        if self.isSquare() and (self.shape[0],1)== v.shape:#checking to see if they can be multiplied in the first place
                                                           # A(n x n) X v(n x 1) = p(n x 1) 
            p = self.matrixMultiply(v)
            plist = p.m
            vlist = v.m
            n = p.shape[0]
            if [0] in vlist or [0] in plist:
                i = vlist.index([0])
                if vlist[i] != plist[i]: # 0 = v[i][0] ---> 0 = p[i][0]
                    return False
            return True
        else:
            return None
    def getEigenvalue(self,v):
        if self.isEigenvector(v):
            p = self.matrixMultiply(v) #product of matrix and column vector v
            plist = p.m
            vlist = v.m
            n = p.shape[0]
            if [0] in vlist or [0] in plist:
                i = vlist.index([0])
                if vlist[i] != plist[i]:
                    return None
            else:
                lam = plist[0][0]/vlist[0][0]
                return lam

length = 3

a = [[-6,3],
     [4,5]]


b = [[3,0,0],
    [0,3,0],
    [0,0,3]]

A = matrix(a)
v1 = [[1],
      [4]]
v1 = matrix(v1)
print(A.isEigenvector(v1))
print2dList(v1)
print("lambda: ",A.getEigenvalue(v1))
print2dList(A.matrixMultiply(v1))
def getEigenvectors(A,Niter=100):
    if A.isSquare():
        d = A.det()
        d = abs(d)
        length = A.shape[0]
        vectorsieve = [] #contains our found eigenvectors
        valuesieve = [] #contains our found eigenvalues
        for i in range(1000):
            #initiallizing our "guess" vector
            b_kList = randVector((0,d),length)
            b_k = matrix(b_kList)
            n = 0    
            while A.isEigenvector(b_k) == False and n<=Niter:
                b_k1 = A.matrixMultiply(b_k)
                maxVal = 0
                for i in range(length):
                    if b_k1.m[i][0]>=maxVal:
                        maxVal = b_k1.m[i][0]
                b_k = b_k1.scalarMultiply(1/maxVal)
                n +=1
            if (A.isEigenvector(b_k)): #and (b_k not in vectorsieve): #and (len(vectorsieve) < length):
                eigenvector = b_k
                eigenvalue = A.getEigenvalue(eigenvector)
                if len(valuesieve) < length: # n-dimensional matrices have at most n eigenvalues
                    valuesieve.append(eigenvalue)
                    vectorsieve.append(eigenvector)
                
        return (valuesieve,vectorsieve)
    else: return None

#b = getEigenvectors(A)
#print(b)
#bval = b[0]
#bvec = b[1]
#for i in range(len(bval)):
#    print2dList(bvec[i])    
#    print("lambda: ",bval[i])
#    print2dList(A.matrixMultiply(bvec[i]))