import itertools
from collections import namedtuple


# ***********************(a)*****************************

Perceptron = namedtuple('Perceptron',['eta','weightColMat','trace'])
Matrix = namedtuple('Matrix',['rows', 'cols', 'data'])
Vector = namedtuple('Vector',['data'])

# ***********************(b)*****************************
def makePerceptron(eta,n,func,trace):
    value = func()
    return Perceptron(eta,Matrix(n+1,1,list(itertools.repeat(value, n+1))),trace)


# ***********************(c)*****************************

def sigma(x):
    if x > 0:
        return 1
    else:
        return 0


# ***********************(d)*****************************

def applyPerceptron(p,augColMat):
    sum = 0
    for i in range(len(p.weightColMat.data)):
        sum += p.weightColMat.data[i]*augColMat.data[i]

    return sigma(sum)





# ***********************(e)*****************************
def applyPerceptronVec(p,vec):

    sum = 0
    vec.data.append(1)

    for i in range(len(p.weightColMat.data)):
        sum += p.weightColMat.data[i] * vec.data[i]

    return sigma(sum)



# ***********************(f)*****************************
def trainOnce(p,inputVec,target):
    sum=0
    inputVec.data.append(1)
    for i in range(len(p.weightColMat.data)):
        sum+= p.weightColMat.data[i] * inputVec.data[i]

    y0 = sigma(sum)

    updatedWeight = []
    for i in range(len(p.weightColMat.data)):
        changeInWt = p.eta * (target - y0) * inputVec.data[i]
        updatedWeight.append(changeInWt)


    if any(updatedWeight):
        for k in range(len(p.weightColMat.data)):
            p.weightColMat.data[k] += updatedWeight[k]
        if p.trace == 2:
            print("On sample ",inputVec,", ",target,", ",p)
        return True
    else:
        return False




# ***********************(g)*****************************
v0 = Vector([1, 1])
v1 = Vector([1, 0])
v2 = Vector([0, 1])
v3 = Vector([0, 0])

andDataSet = [(v0, 1),
(v1, 0),
(v2, 0),
(v3, 0)]

# ***********************(h)*****************************
orDataSet = [(v0, 1),
(v1, 1),
(v2, 1),
(v3, 0)]

# ***********************(i)*****************************
def trainEpoch(p,dataset):
    lst = []
    for inputVec in dataset:
        b = trainOnce(p,inputVec[0],inputVec[1])
        lst.append(b)
    if p.trace == 1:
        print("After epoch: ",p)
    if True in lst:

        return True
    else:
        return False


# ***********************(j)*****************************

def train(p,dataset,bound):
    value = True
    for _ in range(bound):
        if value:
            value = trainEpoch(p,dataset)
        else:
            break


if __name__ == '__main__':
    p = makePerceptron(.2, 2, lambda: .5, 1)
    train(p,orDataSet,10)
