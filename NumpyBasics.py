
# This file shall demonstrate some basics of the Numpy module

# Numpy official tutorials: https://docs.scipy.org/doc/numpy/user/quickstart.html

# Numpy provides a homogenous multidimensional array as it's main data type
# It usually is comprised of numbers. This array is indexed by a tuple of positive numbers

# The dimensions are referred to as axes in Numpy

import numpy as np

# Numpy's array is called a ndarray. It is aliased as an array. This is different from the Python's array module array

# Note that when we print arrays:
# One-dimensional arrays are then printed as rows
# Bidimensionals are printed as matrices
# Tridimensionals are printed as lists of matrices.

def ArrayCreation():

    # Basic single dimension array creation from a list:
    print("Basic single dimension array creation from a list:")
    npArray = np.array([1, 2, 3])
    print(npArray)
    print(npArray.dtype)        # This prints the type of the data of the nparray. Here, it returns int32
    print(npArray.itemsize)     # Prints the size of each element (in bytes) in this array. Here, it prints 4
    print(npArray.size)         # Prints the size of the array. Here, it prints 3
    print(npArray.ndim)         # Prints the dimensions of the array. Here, it prints 1
    print(npArray.shape)        # Prints the dimensios of the array. Here, it prints (3,)
    print(npArray.data)         # Prints the details of the buffer containing the data. Prints something like <memory at 0x000001B51225F588>

    # A single dimension array with floats
    print("\n\nA single dimension array with floats: ")
    npFloatArray = np.array([1.1, 2.2, 3.3])
    print(npFloatArray)
    print(npFloatArray.dtype)  # Prints float64
    print(npFloatArray.itemsize)  # It prints 8  (since the data type is a float64, i.e., 64 bites or 8 bytes)

    # A single dimension array created from a tuple
    print("\n\nA single dimension array created from a tuple: ")
    npArrayFromTuple = np.array(('a', 'b', 'c'))
    print(npArrayFromTuple)
    print(npArrayFromTuple.dtype)  # Prints <U1 representing a 1 byte unsigned integer

    # This produces an error if run
    # npArr = np.array(1,2,3)   # ValueError: only 2 non-keyword arguments accepted

    # Arrays convert sequence of sequences into 2d arrays, sequence of sequence of sequences in 3d arrays and so on
    seqOfSeqList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    npArrSeqOfSeq = np.array(seqOfSeqList)
    print("List of Lists: \n{0}".format(seqOfSeqList))
    print("Array from List of Lists: \n{0}\nShape: {1}, Dimension {2}, Size {3}".format(npArrSeqOfSeq, npArrSeqOfSeq.shape,
                                                                                        npArrSeqOfSeq.ndim, npArrSeqOfSeq.size))
    # Shape: (3, 3), Dimension 2, Size 9

    seqOfSeqOfSeqTuple = (((1,2),(3,4)),((5,6),(7,8)))
    npArrSeqOfSeqOfSeq = np.array(seqOfSeqOfSeqTuple)

    print("Tuple of Tuples of Tuples: \n{0}".format(seqOfSeqOfSeqTuple))
    print("Array from Tuple of Tuples of Tuples: \n{0}\nShape: {1}, Dimension {2}, Size {3}".format(npArrSeqOfSeqOfSeq,
                                                                                        npArrSeqOfSeqOfSeq.shape,
                                                                                        npArrSeqOfSeqOfSeq.ndim,
                                                                                        npArrSeqOfSeqOfSeq.size))
    # Shape: (2, 2, 2), Dimension 3, Size 8

    # Creating arrays from heterogenous lists lead to all the elements getting converted to one single type
    print("\n\nArrays created from heterogenous lists:")
    npHeterogenousArr = np.array([1, 'b', 1.5])
    print(npHeterogenousArr)  # everything here becomes a string: ['1' 'b' '1.5']
    print(npHeterogenousArr.dtype)  # <U11
    npHeterogenousArr = np.array([1, True, 1.5])
    print(npHeterogenousArr)  # everything here becomes a float: [1.  1.  1.5]
    print(npHeterogenousArr.dtype)  # float64

    # Specifying the data type of the array elements at the creation time:
    print("\n\nArrays created by specifying the data type at the creation time:")
    npStringArr = np.array([1,2,3], dtype = str)
    print(npStringArr)
    npFloatArray = np.array((1,2,3), dtype=float)
    print(npFloatArray)
    # If we specify a type that cannot accomodate the values passed in, we get an error:
    # ValueError: invalid literal for int() with base 10: 'a'
    # npTestArr = np.array(['a', 1], dtype=int)
    # print(npTestArr)

    # Array expansion operations are expensive. Hence, if we have a scenario where we know the array size in advance,
    # we can then initialize an array of that specified size and have it initialized with zeroes or ones or some random numbers:
    print("\n\nCreating blank arrays: ")
    # The first param, shape, can be a tuple or a list, specifying the dimensions of the array
    # The second param, dtype, specifies the type of the array elements. Default value is float
    # This has a third parameter called order, which can be used to specify a row major or a column major order
    # The params are either 'C' or 'F' for either C style or Fortran style. We can leave as the default value ('C')
    npZeroArr = np.zeros(shape = (3,4), dtype=int)
    print(npZeroArr)
    npOnesArr = np.ones([2,3], dtype=str)
    print(npOnesArr)
    npEmptyArr = np.empty((2,3))
    print(npEmptyArr)  # Prints some random numbers that are present in the memory where this is created

    # Creating arrays using arange:
    # numpy provides the arange function that is similar to the range function provided by python
    # Instead of returing a list like the range does, arange returns an array
    # Signature: numpy.arange([start, ]stop, [step, ]dtype=None)
    # Default start value is 0. Default step is 1
    print("\n\nUsing the arange to generate arrays:")
    print(np.arange(10, 50, 2))  # Prints [10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48]
    print(np.arange(-5, 5, 1.2))  # Prints [-5.  -3.8 -2.6 -1.4 -0.2  1.   2.2  3.4  4.6]
    print(np.arange(5))  # Prints [0 1 2 3 4] since 5 is taken as the mandatory parameter, stop
    # print(np.arange('a', 'z', 1))  # Error: TypeError: unsupported operand type(s) for -: 'str' and 'str'

    # If we want to create an array for floats, it is better to use linspace instead of arange due to the
    # unpredictablility of using floats
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    # From the documentation:
    # Return evenly spaced numbers over a specified interval.    #
    # Returns num evenly spaced samples, calculated over the interval [start, stop].    #
    # The endpoint of the interval can optionally be excluded.

    print("\n\nUsing linspace to generate float arrays:")
    print(np.linspace(start=-5, stop=7, num=11))  # [-5.  -3.8 -2.6 -1.4 -0.2  1.   2.2  3.4  4.6  5.8  7. ]

    # There are some functions that can create arrays of the same dimensions as the existing ones:
    npSampleArr = np.array([[1,2,3],[4,5,6]])
    print("Template array is {0}".format(npSampleArr))
    print("Zeros array using zeros_like: {0}".format(np.zeros_like(npSampleArr)))
    print("Ones array using ones_like: {0}".format(np.ones_like(npSampleArr)))
    print("Empty array using empty_like: {0}".format(np.empty_like(npSampleArr)))
    print("Using full_like to initialize an array of same size as an existing one but with a default value:")
    # If the fill_value can't match the original array type, we need to specify the destination array type
    # Else, we will get an error
    print(np.full_like(npSampleArr, fill_value='a', dtype=str))

    # If an array is too big to print, only the head and tail sections are printed out
    npLargeArr = np.arange(10000)
    print(npLargeArr)  # Printed [   0    1    2 ... 9997 9998 9999]
    npLargeArr = npLargeArr.reshape(5, 2000)
    print(npLargeArr)
    # The above printed:
    # [[0    1    2... 1997 1998 1999]
    # [2000 2001 2002... 3997 3998 3999]
    # [4000 4001 4002 ... 5997 5998 5999]
    # [6000 6001 6002 ... 7997 7998 7999]
    # [8000 8001 8002 ... 9997 9998 9999]]

    #  To force printing the entire array, set np.set_printoptions(threshold=np.nan)

    # Creating an array from using the random object:
    # numpy.random.rand(d0, d1, ..., dn)
    #   -> Create an array of the given shape and populate it with random
    #      samples from a uniform distribution over [0, 1).
    print(np.random.rand(3, 2))
    # Example output:
    # [[0.96998764 0.29951753]
    #  [0.18584147 0.13551683]
    #  [0.49355695 0.23795125]]

    # numpy.random.randn(d0, d1, ..., dn)
    #   -> Return a sample (or samples) from the “standard normal” distribution.
    print(np.random.randn(3, 2))
    # Example output:
    #  [[-0.59547268, -1.2990651],
    #   [-0.8873276, -0.57430988],
    #   [0.75189181, 1.54554595]]

def ArrayShapeManipulation():
    # We can use the reshape function to alter the dimensions of an existing array

    np1dArray = np.arange(10)
    print("Existing array {0} is of shape {1}".format(np1dArray, np1dArray.shape))  # [0 1 2 3 4 5 6 7 8 9], shape (10,)
    npReshapedArr = np1dArray.reshape(2,5)
    print("Reshaped array {0} is of shape {1}".format(npReshapedArr, npReshapedArr.shape))
    # The original array is converted to :
    #[[0 1 2 3 4]
    # [5 6 7 8 9]]
    # Shape: shape (2, 5)

    npFlattenedArr = np.ravel(npReshapedArr)
    print("Flattened array : {0}".format(npFlattenedArr))  # [0 1 2 3 4 5 6 7 8 9]
    # Transpose of a matrix is the matrix whose rows are the columns of the original matrix and
    # whose columns are the rows of the original matrix
    # Thus, if the original matrix is :
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]
    # then the transposed matrix will be:
    # [[1, 3, 5],
    #  [2, 4, 6]]
    # To obtain the transpose of an array:
    npTempArr = np.array([[1,2],[3,4],[5,6]])
    print("The transpose of the array \n{0} \nis \n{1}".format(npTempArr, npTempArr.T))
    # The transpose of the array
    # [[1 2]
    #  [3 4]
    #  [5 6]]
    # is
    # [[1 3 5]
    #  [2 4 6]]

    # Reshape, Ravel and Transpose do not modify the original array.
    # Resize does
    # numpy.resize(a, new_shape)[source]
    npTempArr = np.arange(6)
    print("Resize altered the array \n{0} to \n{1}".format(npTempArr, np.resize(npTempArr,(3,2))))
    # Note the usage of the resize function. It is a method of the np object, not the array object

def ArrayBasicOperations():
    # Operations on the arrays are performed elementwise. This results in a new array, the original is unaffected

    # When operating with arrays of different types, the type of the resulting array corresponds to the more general
    # or precise one (a behavior known as upcasting).
    # If such a casting cannot happen, an error is thrown. For e.g., if we operate between an array of strings and
    # an array of integers
    # a = np.array(['a','b'])
    # b = np.array([1,2])
    # print(a + b)
    # TypeError: ufunc 'add' did not contain a loop with signature matching types dtype('<U11') dtype('<U11')
    # dtype('<U11')

    npTempArr = np.arange(1, 11)
    print(npTempArr)      # [ 1  2  3  4  5  6  7  8  9 10]
    print(npTempArr + 1)  # [ 2  3  4  5  6  7  8  9 10 11]
    print(npTempArr - 2)  # [-1  0  1  2  3  4  5  6  7  8]
    print(npTempArr * 3)  # [ 3  6  9 12 15 18 21 24 27 30]
    print(npTempArr / 2)  # [0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]

    # To perform these operations in place, use the += operator, -= operator, etc.
    npTempArr += 3
    print(npTempArr)   # npTempArr is now changed to [ 3  6  9 12 15 18 21 24 27 30]

    # Testing divide by zero
    npTempArr = np.arange(-1, 5)
    print(npTempArr)      # [-1  0  1  2  3  4]
    print(npTempArr / 0)  # RuntimeWarning: invalid value encountered in true_divide print(npTempArr / 0)
    # Prints [-inf  nan  inf  inf  inf  inf]

    # Operations between arrays:
    # This results in a new array whose elements are [arr1]operand[arr2]
    npArr1 = np.arange(-5, 5)
    npArr2 = np.arange(0, 10)

    print("Arrays are :\n{0} \nand \n{1}\n".format(npArr1, npArr2))
    print("Adding them:")
    print(npArr1 + npArr2)      # [-5 -3 -1  1  3  5  7  9 11 13]
    print("Subtracting them:")
    print(npArr1 - npArr2)      # [-5 -5 -5 -5 -5 -5 -5 -5 -5 -5]
    print("Multiplying them:")
    print(npArr1 * npArr2)      # [ 0 -4 -6 -6 -4  0  6 14 24 36]
    print("Dividing them:")
    print(npArr1 / npArr2)      # [ -inf -4. -1.5 -0.66666667 -0.25 0. 0.16666667 0.28571429 0.375 0.44444444]

    # In place operation:
    npArr1 += npArr2
    print("\n\nnpArr1 after an inplace operation, npArr1 += npArr2:")
    print(npArr1)

    # To produce actual matrix multiplication, we need to use the @ operator or the .dot function:
    print("Matrix multiplcation:")
    npArr1 = np.array([[1,2],[3,4],[5,6]])
    npArr2 = np.array([[10,20],[30,40]])

    print("The arrays are \n{0}\nand\n{1}".format(npArr1, npArr2))

    # [[1 2]
    #  [3 4]
    #  [5 6]]

    # [[10 20]
    #  [30 40]]

    print("Result of matrix multiplication:")
    print(npArr1 @ npArr2)
    print(npArr1.dot(npArr2))

    # Both produce:
    # [[70 100]
    #  [150 220]
    #  [230 340]]

    # Some useful functions

    npTempArr = np.random.rand(3,4)
    # Finding the sum of all the elements in the array:
    print("Sum of elements of \n{0}\nis\n{1}".format(npTempArr, npTempArr.sum()))
    print("Min element is {0} and max element is {1}".format(npTempArr.min(), npTempArr.max()))

    # We can specify the axis along which we need to operate to get an array of outputs instead of a single value:
    print("Sum of elements along each rows:")
    print(npTempArr.sum(axis=1))
    print("Sum of elements along each column:")
    print(npTempArr.sum(axis=0))

    # Similarly, we can find the min and max for each row or column:

    print("Min element for each row is: \n{0}".format(npTempArr.min(axis=1)))
    print("Min element for each column is: \n{0}".format(npTempArr.min(axis=0)))
    print("Max element for each row is: \n{0}".format(npTempArr.min(axis=1)))
    print("Max element for each column is: \n{0}".format(npTempArr.min(axis=0)))

    # TODO: Demonstrate using universal functions like sin, cos, etc. here:

def Main():

   print("Numpy Array Basics")
   ArrayCreation()
   ArrayShapeManipulation()
   ArrayBasicOperations()



if __name__ == "__main__":
    Main()