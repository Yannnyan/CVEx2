import math
import numpy as np
import cv2

def myID() -> np.int32:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 323906842

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    flipped_kernel = np.flip(k_size)
    k = flipped_kernel.shape[0]
    n = in_signal.shape[0]
    result_size = in_signal.shape[0] + flipped_kernel.shape[0] - 1
    result = np.zeros((result_size))
    for i in range(result_size):
        # lk and rk are the kernel indices
        lk = np.max([0, k-i-1])
        rk = k if i < n else k - (i-n + 1)
        # lsig and rsig are the signal indices
        lsig = np.max([i-k+1,0])
        rsig = i+1 if i < n else n
        # print("kernel index: " + str(lk), str(rk), "signal index: " + str(lsig), str(rsig)) - for debugging
        result[i] = np.sum(flipped_kernel[lk:rk] * in_signal[lsig:rsig])
    return result


def getPaddedArray(in_image: np.ndarray,kernel: np.ndarray) -> np.ndarray:
    n_y,n_x = in_image.shape
    k_y,k_x = kernel.shape
    added_rows = np.int32(np.floor(kernel.shape[0]/2))
    added_cols = np.int32(np.floor(kernel.shape[1]/2))
    # get padded matrix
    im_pad = np.zeros((n_y + added_rows*2, n_x + added_cols*2))
    im_pad[added_rows: added_rows + n_y, added_cols: added_cols + n_x] = in_image
    # im_pad = np.pad(in_image,(added_rows,added_cols), "constant")
    # init the upper and lower border
    im_pad[0:added_rows,:] = im_pad[added_rows : 2 * added_rows,:]
    im_pad[n_y + added_rows : (n_y + 2 * added_rows), :] = im_pad[n_y : (n_y + added_rows), :]
    # # init left and right borders
    im_pad[:,0:added_cols] = im_pad[:, added_cols:2*added_cols]
    im_pad[:,(n_x + added_cols): (n_x + 2 * added_cols)] = im_pad[:, n_x : n_x + added_cols]
    return im_pad


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    result = np.zeros(in_image.shape)
    k_y,k_x = kernel.shape
    r_y,r_x = result.shape
    # im_pad = getPaddedArray(in_image,kernel)
    im_pad = cv2.copyMakeBorder(in_image, k_y // 2, k_y // 2, k_x // 2, k_x // 2, cv2.BORDER_REPLICATE)
    flipped_kernel = np.flip(kernel)
    # got over the y axis with stride 1
    for row in range(0,r_y,1):
        # go over in the x axis with stride 1
        for col in range(0,r_x,1):
            # convolve the kernel
            result[row,col] = np.sum(im_pad[row: row + k_y, col:col + k_x] * flipped_kernel)
    return result


def convDerivative(in_image: np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel_y = np.array([[1],
                       [0],
                       [-1]])
    kernel_x = np.array([[1,0,-1]])
    i_x = cv2.filter2D(in_image,-1,kernel_x)
    i_y = cv2.filter2D(in_image,-1,kernel_y)
    magnitudes = np.sqrt(np.power(i_x,2) + np.power(i_y,2))
    directions = np.arctan2(i_y, i_x)
    return (directions,magnitudes)


def getGaussianFilter(k_size):
    std = 1
    kernel = np.zeros((k_size,k_size))
    constant = (1 / (2 * np.pi * std ** 2)) 
    for s,v in np.ndenumerate(kernel):
        ix,iy = s[0],s[1]
        power_param = -(ix ** 2 + iy **2) / 2 * std**2
        kernel[ix,iy] = constant * np.e ** power_param
    return kernel

def getBinomialFilter(k_size):
    vector_size = k_size + (k_size // 2) * 2
    kernel = np.zeros(vector_size)
    #[0.. 0 1 0 .. 0]
    kernel[(0)] = 1
    # [1 1]
    filter_iden = np.array([1, 1])
    filter_return = np.zeros((k_size,k_size))
    for i in range(0,k_size-1,1):
        kernel = np.convolve(kernel, filter_iden,'same')
    index_put = 0
    # we achieved the coeffiecients to start from
    for i in range(0,k_size//2+1,1):
        filter_return[index_put,:] = np.copy(kernel[index_put: index_put + k_size])
        filter_return[-index_put-1,:] = np.copy(kernel[index_put: index_put + k_size])
        for j in range(2):
            kernel = np.convolve(kernel,filter_iden,'same')
        index_put+=1
    return filter_return

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # kernel = getGaussianFilter(k_size)
    kernel = getBinomialFilter(k_size)
    kernel =  kernel / np.sum(kernel)
    filtered_image = cv2.filter2D(in_image,-1,kernel)
    return filtered_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    return cv2.GaussianBlur(in_image, (k_size,k_size),cv2.BORDER_REPLICATE) 


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
