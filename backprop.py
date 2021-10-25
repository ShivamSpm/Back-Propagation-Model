"""
Name: Shivam Mahajan
id: spm9398

"""

from random import random

from linearAlgebra import *

BackProp = namedtuple('BackProp',['eta','weightMats','trace'])


def makeBackProp(eta,n,thunk,trace):

    matrixList=[]
    for i in range(len(n)-1):
        if i+1 == len(n)-1:
            m = makeMatrix(n[i + 1], n[i] + 1, thunk)

        else:
            m = makeMatrix(n[i+1]+1,n[i]+1,thunk)

        matrixList.append(m)

    return BackProp(eta,matrixList,trace)


def sigma(x):
    e = math.exp(-x)
    result = 1/(1+e)
    return result


def applyBackPropAll(B,augColumnMat):
    matrixList = []
    matrixList.append(augColumnMat)
    for i in range(len(B.weightMats)):
        m = mult(B.weightMats[i],matrixList[i])
        for k in range(len(m.data)):
            m.data[k] = sigma(m.data[k])

        matrixList.append(m)

    return matrixList

andOrB = makeBackProp(0.2, [2, 2], (lambda: random()-.5), 0)
xorB =  makeBackProp(0.2, [2, 2, 1], (lambda: random()), 0)

def applyBackPropVec(B,nonAugVector):
    nonAugVector.data.append(1)
    m = Matrix(len(nonAugVector.data),1,nonAugVector.data)
    resultList = applyBackPropAll(B,m)
    resultVector = Vector(resultList[len(resultList)-1].data)
    return resultVector

def trainOnce(B,inputVec,targetOutputVec):

    if(len(inputVec.data)!=B.weightMats[0].cols):
        inputVec.data.append(1)

    outputMatrixList = applyBackPropAll(B, Matrix(len(inputVec.data), 1, inputVec.data))

    lossBYw = []

    for i in range(len(B.weightMats),0,-1):
        xBYw = outputMatrixList[i-1]
        oneMat = makeMatrix(len(outputMatrixList[i].data),1, lambda: 1)
        yBYx = pointProd(outputMatrixList[i], substract(oneMat, outputMatrixList[i]))
        targetOutputMat = Matrix(len(targetOutputVec.data),1,targetOutputVec.data)
        if(i==len(B.weightMats)):

            loss = substract(outputMatrixList[i],targetOutputMat)
            lossBYx = pointProd(loss,yBYx)

            lossBYw.append(scale(-B.eta,outerProd(lossBYx,xBYw)))
        else:

            wMat = transpose(B.weightMats[i])

            lossBYy = mult(wMat,lossBYx)
            lossBYx = pointProd(lossBYy,yBYx)
            lossBYw.append(scale(-B.eta,outerProd(lossBYx,xBYw)))
    updatedW = []
    lossBYw.reverse()
    for i in range(len(lossBYw)):
        B.weightMats[i] = add(lossBYw[i],B.weightMats[i])

    if(B.trace==2):
        print("On sample: ","Input vector: ",inputVec,"\nOutput vector: ",targetOutputVec,"\nBack-propagation object: ",B)



andOrDataSet = [(Vector([1, 1]),Vector([1, 1])),(Vector([1, 0]),Vector([0, 1])),(Vector([0, 1]),Vector([0, 1])),(Vector([0, 0]),Vector([0, 0]))]
xorDataSet = [(Vector([1,1]),Vector([0])),
(Vector([1,0]),Vector([1])),
(Vector([0,1]),Vector([1])),
(Vector([0,0]),Vector([0]))]

def trainEpoch(B,dataset):
    if B.trace==1:
        print("After epoch: ",B)
    for i in range(len(dataset)):
        trainOnce(B,dataset[i][0],dataset[i][1])


def train(B,dataset,n):

    for i in range(n):
        trainEpoch(B,dataset)




def main():


    train(xorB,xorDataSet,5000)
    train(andOrB,andOrDataSet,5000)
    print("After Training(andOr): ")
    print(applyBackPropVec(andOrB,Vector([1,1])))
    print(applyBackPropVec(andOrB,Vector([0,1])))
    print(applyBackPropVec(andOrB,Vector([1,0])))
    print(applyBackPropVec(andOrB,Vector([0,0])))
    print("Weights:",andOrB.weightMats)
    print("----------------------")
    print("\nAfter Training(xor): ")
    print(applyBackPropVec(xorB,Vector([1,1])))
    print(applyBackPropVec(xorB,Vector([0,1])))
    print(applyBackPropVec(xorB,Vector([1,0])))
    print(applyBackPropVec(xorB,Vector([0,0])))
    print("Weights:",xorB.weightMats)

if __name__ == '__main__':
    main()
