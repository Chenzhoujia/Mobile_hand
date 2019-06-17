import numpy as np


def getPositon():
    a = np.mat([[2, 5, 7, 8, 9, 89], [6, 7, 5, 4, 6, 4]])

    raw, column = a.shape  # get the matrix of a raw and column

    _positon = np.argmax(a)  # get the index of max in the a
    print(_positon)
    m, n = divmod(_positon, column)
    print("The raw is ", m)
    print("The column is ", n)
    print("The max of the a is ", a[m, n])


getPositon()