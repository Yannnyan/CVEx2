import unittest
from ex2_utils import * 

class tests(unittest.TestCase):
#     def test(self):
#         arr = [1, 1]
#         arr1 = cv2.imread('boxMan.jpg',cv2.IMREAD_GRAYSCALE)
        # print(cv2.filter2D(arr1,-1,np.array(arr)))

    def testBlur(self):
        # print(getGaussianFilter(3))
        print(getBinomialFilter(5))
        print(cv2.getGaussianKernel(5,1/273))
        

#     def testConv1d(self):
#         arr1 = [1,2,3,4,5]
#         arr2 = [1,2,3,4]
#         result = conv1D(np.array(arr1),np.array(arr2))
#         # result = []
#         expected = np.convolve(arr1,arr2)
#         # self.assertTrue(np.equal(result,expected))
#         print("result: ", result)
#         print("expected: ", expected)
#         self.assertTrue(np.alltrue(result == expected))
#         arr3 = [1,2]
#         expected1 = np.convolve(arr1,arr3)
#         result1 = conv1D(np.array(arr1), np.array(arr3))
#         self.assertTrue(np.alltrue(expected1 == result1))
#         print("result: ", result1)
#         print("expected: ", expected1)
#         print("----------------------------")

#     def testConv2d(self):
#         arr1 = cv2.imread('boxMan.jpg',cv2.IMREAD_GRAYSCALE)
#         # print(arr1, arr1.shape)
#         arr2 = [[-1,-1,-1],
#                 [-1,9,-1],
#                 [-1,-1,-1]]
#         result = conv2D(np.array(arr1),np.array(arr2)).astype(np.float64)
#         expected = cv2.filter2D(src=arr1,ddepth=-1,kernel=np.array(arr2),borderType=cv2.BORDER_REPLICATE).astype(np.float64)
#         print("result: ", result,result.shape)
#         print("expected: ", expected, expected.shape)
#         self.assertTrue(np.alltrue(result == expected))
#         print("---")
#         # test2
#         arr3 = [[1],
#                 [0],
#                 [-1]]
#         result1 = conv2D(np.array(arr1), np.array(arr3))
#         expected1 = cv2.filter2D(src=arr1,ddepth=-1,kernel=np.array(arr3),borderType=cv2.BORDER_REPLICATE)
#         print("result1", result1, result1.shape)
#         print("expected1", expected1, expected1.shape)
#         self.assertTrue(np.alltrue(result1 == expected1))

#     def testPadding(self):
#         arr1 = np.ones((4,4))
#         arr2 = [[-1,-1,-1],
#                 [-1,9,1],
#                 [3,-1,3]]
#         arr_expected = [[ 7 , 7 , 2 , 2 , 9 , 9],
#                         [ 7 , 7 , 2 , 2 , 9 , 9],
#                         [-1 ,-1 , 1 , 1 , 1 , 1],
#                         [-1 ,-1 , 1 , 1 , 1 , 1],
#                         [ 8 , 8 , 3 , 3 , 6 , 6],
#                         [ 8 , 8 , 3 , 3 , 6 , 6]]
#         arr1[:,0] = np.array([-1,-1,-1,-1])
#         arr1[0,:] = np.array([7,2,2,9])
#         arr1[arr1.shape[0]-1,:] = np.array([8,3,3,6])
#         # print(arr1, arr1.shape)
#         padded = getPaddedArray(arr1,np.array(arr2))
#         # print(padded,padded.shape)
#         self.assertTrue(np.alltrue(np.array(arr_expected) == padded))
#         arr3 = [[1],
#                 [0],
#                 [-1]]
#         padded1 = getPaddedArray(arr1, np.array(arr3))
#         print(padded1, padded1.shape)




        

        

if __name__ == '__main__':
    unittest.main()





