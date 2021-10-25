"""

Name: Shivam Mahajan
id: spm9398

"""

#******************(Q-1)*********************
#******************(a)*********************
import inspect
import itertools
import math
from collections import namedtuple
import matplotlib.pyplot as plt
Vector = namedtuple('Vector',['data'])
# v1 = Vector([4,5,6])

#******************(b)*********************
def makeVector(n, f):
    value = f()
    return Vector(list(itertools.repeat(value, n)))
# print(makeVector(3, lambda : 0.5 ))


#******************(c)*********************
def setVec(v1,v2):
    l = v2.data
    v1.data.clear()
    for i in range(len(l)):
        v1.data.append(l[i])



#******************(d)*********************

Matrix = namedtuple('Matrix',['rows', 'cols', 'data'])


#******************(e)*********************
def makeMatrix(r,c,function):

    data = []
    for i in range(r*c):
        data.append(function())
    return Matrix(r,c,data)

# print(makeMatrix(3,3,lambda :0.5))


#******************(f)*********************


def setMat(m1,m2):
    if not (m1.rows==m2.rows and m1.cols==m2.cols):
        print( "Incompatible rows and columns")


    m = m2.data
    m1.data.clear()


    for i in range(len(m)):
        m1.data.append(m[i])

# m1 = Matrix(2,3,[1,3,2,7,5,8])
# m2 = Matrix(2,3,[1,2,4,5,6,9])




#******************(g)*********************

def augmentColMat(colMat):
    colMat.data.append(1)
    return Matrix(len(colMat.data),colMat.cols,colMat.data)


#******************(h)*********************


def colMatrixFromVector(vector):

    return Matrix(len(vector.data),1,vector.data)


#******************(i)*********************


def vectorFromColMatrix(colMat):
    return Vector(colMat.data)




#******************(j)*********************

def mapMatrix(f, A):
    l=[]
    for x in A.data:
        l.append(f(x))
    return Matrix(A.rows,A.cols,l)




#******************(k)*********************

def add(a,b):
    if not type(a) == type(b):
        raise TypeError("Incompatible types for addition")
    else:
        if isinstance(a,int) :
            return a+b
        elif isinstance(a,float):
            return a+b

        elif isinstance(a,Vector) :
            if not len(a.data) == len(b.data):
                raise TypeError("Incompatible Vector dimensions")

            l=[]
            for i in range(len(a.data)):
                l.append(a.data[i]+b.data[i])
            return Vector(l)

        elif isinstance(a,Matrix):
            if not a.rows ==b.rows and a.cols==b.cols:
                raise TypeError("Incompatible rows and columns")
            matList = []
            for i in range(len(a.data)):
                matList.append(a.data[i]+b.data[i])
            return Matrix(a.rows,a.cols,matList)



#******************(l)*********************

def substract(a,b):
    if not type(a) == type(b):
        raise TypeError("Incompatible types for substraction")
    if isinstance(a,int) :
        return(a-b)
    elif isinstance(a,float) :
        return(a-b)

    elif isinstance(a,Vector) :
        if not len(a.data) == len(b.data):
            raise TypeError("Incompatible Vector dimensions")
        l=[]
        for i in range(len(a.data)):
            l.append(a.data[i]-b.data[i])
        return (Vector(l))

    elif isinstance(a,Matrix) :
        if not a.rows == b.rows and a.cols == b.cols:
            raise TypeError("Incompatible rows and columns")
        matList = []
        for i in range(len(a.data)):
            matList.append(a.data[i]-b.data[i])
        return (Matrix(a.rows,a.cols,matList))



#******************(m)*********************

def scale(number, b):

    if isinstance(b,int) or isinstance(b,float):

        return(number*b)

    elif isinstance(b,Vector):
        l=[]

        for i in range(len(b.data)):
            # print(b.data[i])
            l.append(b.data[i]*number)
        return(Vector(l))

    elif isinstance(b,Matrix):
        l=[]
        for i in range(len(b.data)):
            l.append(b.data[i] * number)
        return(Matrix(b.rows,b.cols,l))



#******************(n)*********************

def mult(m1,m2):
    if not m1.cols == m2.rows:
        raise TypeError("Incompatible Matrices")


    list1 = [m1.data[r * m1.cols:(r + 1) * m1.cols] for r in range(0, m1.rows)]
    list2 = [m2.data[r * m2.cols:(r + 1) * m2.cols] for r in range(0, m2.rows)]
    sumList = []
    total = 0
    for i in range(m1.rows):
        sumList.append([0 for i in range(len(list2[0]))])


    for i in range(len(list1)):
        for j in range(len(list2[0])):
            for k in range(len(list2)):
                sumList[i][j] += list1[i][k]*list2[k][j]

    matrixList = [itr for l in sumList for itr in l]
    return Matrix(m1.rows,m2.cols,matrixList)




# ******************(o)*********************

def pointProd(a,b):
    if not type(a) == type(b):
        raise TypeError("Incompatible types for point product")

    if isinstance(a, int) or isinstance(b, int):
        return(a*b)
    elif isinstance(a,float) or isinstance(b,float) :
        return(a*b)

    elif isinstance(a,Vector) or isinstance(b,Vector) :
        if not len(a.data) == len(b.data):
            raise TypeError("Incompatible Vector dimensions")
        l=[]
        for i in range(len(a.data)):
            l.append(a.data[i]*b.data[i])
        return(Vector(l))

    elif isinstance(a,Matrix) or isinstance(b,Matrix) :
        if not a.rows == b.rows and a.cols == b.cols:
            raise TypeError("Incompatible rows and columns")
        matList = []
        for i in range(len(a.data)):
            matList.append(a.data[i]*b.data[i])
        return Matrix(a.rows,a.cols,matList)



# ******************(p)*********************

def transpose(m):
    i=0
    j=0
    k=0
    l = []
    while i<m.cols:
        j=0
        while j < m.rows:
            l.append(m.data[k])
            k+=m.cols
            j+=1
        i+=1
        k=i
    return Matrix(m.cols,m.rows,l)


# ******************(q)*********************

def dot(a,b):
    if not type(a) == type(b):
        raise TypeError("Incompatible types")
    if isinstance(a, int) or isinstance(a, float):
        return(a*b)

    if isinstance(a,Vector):
        if not len(a.data) == len(b.data):
            raise TypeError("Incompatible Vector dimensions")
        sum=0
        for i in range(len(a.data)):
            sum+=a.data[i]*b.data[i]
        return(sum)
    if isinstance(a,Matrix):
        if not a.rows == b.rows and a.cols == b.cols:
            raise TypeError("Incompatible matrices")
        m1 = transpose(a)
        l = []
        sum=0
        for i in range(len(m1.data)):
            sum += m1.data[i]*b.data[i]
        return Matrix(m1.rows,b.cols,sum)






# ******************(r)*********************

def outerProd(a,b):
    if not type(a) == type(b):
        raise TypeError("Incompatible types")
    if isinstance(a, int) or isinstance(a, float):
        return(a * b)

    if isinstance(a,Vector):

        m1 = Matrix(len(a.data),1,a.data)
        m2 = Matrix(len(b.data), 1, b.data)
        m = transpose(m2)
        m3 = mult(m1,m)

        return m3


    if isinstance(a,Matrix):

        m = transpose(b)
        matrix = mult(a,m)
        return matrix


# ******************(Q-2)*********************

# ******************(a)*********************

def euler1(h,tb,x0,F):

    x = x0
    t = 0
    while t<=tb:
        print(t,x)
        x = x + h*F(x,t)
        t = t + h





# ******************(b)*********************

T = []
X = []

def update(t, x):
  global T, X
  T.append(t)
  X.append(x)

def euler2(h,tb,A,x0,F):
    x = x0
    t = 0
    while t <= tb:
        A(t,x)
        x = x + h * F(x, t)
        t = t + h




# ******************(c)*********************

def euler(h,tb,A,x0,F):

    t = 0
    x = x0

    while t <= tb:
        l = []
        for i in range(len(F)):
            l.append(x[i])

            x[i] = x[i] + h * F[i](x, t)
        A(t, l)
        t = t + h







# ******************(d)*********************
def sin():
    euler(0.0025,100,update, [0,1], [lambda x, t: x[1],lambda x, t: -x[0]])
    sinList = []
    for x in X:
        sinList.append(x[0])
    plt.plot(T,sinList)
    plt.show()



# ******************(e)*********************
def eulerCurry(h):
    def tb(tb):
        def A(A):
            def x0(x0):
                def F(F):
                    t=0
                    x = x0
                    while t <= tb:
                        l = []
                        for i in range(len(F)):
                            l.append(x[i])

                            x[i] = x[i] + h * F[i](x, t)
                        A(t, l)
                        t = t + h

                return F
            return x0
        return A
    return tb





# ******************(Q-3)*********************

#Plot of question 3 with T=0.5
def q_3a():

    a = 0.7
    b = 0.8
    tau = 12.5
    euler(0.0025,200 ,update, [1,1.2], [lambda x, t: (x[0] - (x[0]*x[0]*x[0]/3) - x[1] + 0.5 ),lambda x, t:(x[0]+a - b*x[1])/tau ])
    q3_List = []
    for x in X:
        q3_List.append(x[0])
    plt.plot(T, q3_List)
    plt.show()


#Plot of question 3 with T=0.5*sin(t)
def q_3b():

    a = 0.7
    b = 0.8
    tau = 12.5
    euler(0.0025,200,update, [1,1.2], [lambda x, t: (x[0] - (x[0]*x[0]*x[0]/3) - x[1] + 0.5*math.sin(t) ),lambda x, t:(x[0]+a - b*x[1])/tau ])
    q3_List = []
    for x in X:
        q3_List.append(x[0])
    plt.plot(T, q3_List)
    plt.show()





