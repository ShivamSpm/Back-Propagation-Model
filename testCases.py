from backprop import *
import inspect



def question1():
    if (len(inspect.signature(BackProp).parameters) != 3):
        print("Error in BackProp creation")

    b = makeBackProp(0.1, [2, 2, 1], lambda: 0.1, 1)
    if (b !=
            BackProp(eta=0.1, weightMats=[Matrix(rows=3, cols=3, data=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                          Matrix(rows=1, cols=3, data=[0.1, 0.1, 0.1])], trace=1)):
        print("Error in makeBackProp function")

    if ((sigma(1) != 0.7310585786300049) or (sigma(-4) != 0.01798620996209156)):
        print("Error in sigma function")

    d1 = applyBackPropAll(b, Matrix(rows=3, cols=1, data=[0, 1, 1]))
    d2 = [Matrix(rows=3, cols=1, data=[0, 1, 1.0]),
          Matrix(rows=3, cols=1, data=[0.549833997312478, 0.549833997312478, 0.549833997312478]),
          Matrix(rows=1, cols=1, data=[0.5411443022794791])]

    if (d1 != d2):
        print("Error in applyBackPropAll function")

    if (applyBackPropVec(b, Vector([1,1])) != Vector(data=[0.5429768786870424])):
        print("Error in applyBackProp function")



def main():
    question1()



if __name__ == '__main__':
    main()