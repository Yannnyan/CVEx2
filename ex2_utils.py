import math
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

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
    added_rows = np.int32(np.floor(k_y/2))
    added_cols = np.int32(np.floor(k_x/2))
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

def gaussianKernel(size, sigma, twoDimensional=True):
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)

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
    kernel_blur_size = 15
    img_blurred = blurImage2(img,kernel_blur_size)
    kernel_size = 15
    dst = cv2.Laplacian(img_blurred, -1, ksize=kernel_size)
    minimum = np.min(dst)
    maximum = np.max(dst)
    dst = (dst + np.abs(minimum)) / (np.abs(maximum) + np.abs(minimum))
    dst = dst >= 0.5
    return dst


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return

def somecircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m+R_max,x+m+R_max]                                           #Computing the extreme X values
            Y= [y-n+R_max,y+n+R_max]                                            #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,R_max:-R_max,R_max:-R_max].tolist()

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
    candidate_circles = []
    max_circles = 20
    min_radius =  min_radius if min_radius > 10 else 10
    max_radius = max_radius if max_radius > 20 else max_radius + 10
    for r in range(min_radius,max_radius,1):
        for theta in range(360):
            x,y = np.int32(r*np.cos(theta)), np.int32(r*np.sin(theta))
            candidate_circles.append((x,y,r))
    cannied_img = cv2.Canny((img*255).astype(np.uint8),50,255)
    valid_indices = np.argwhere(cannied_img == 255)
    d = defaultdict(lambda: 0)
    for a,b in valid_indices:
        for (x,y,r) in candidate_circles:
            # check if the center is out of bounds
            if (a+ y) < 0 or (a + y) >= cannied_img.shape[0] \
                or (b + x) < 0 or (b + x) >= cannied_img.shape[1]:
                continue
            d[(b+x,a+y,r)] += 1
    lst = list(map(lambda x: x[0], sorted(d.items(),key=lambda x: x[1],reverse=True)))
    return lst[:max_circles]


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    result = np.zeros(in_image.shape)
    bil_cv = cv2.bilateralFilter(in_image,k_size,sigmaColor=sigma_color,sigmaSpace=sigma_space)
    sig_c_sq = 2* (sigma_color ** 2) # 2 * sigma color squared
    sig_s_sq = 2* (sigma_space ** 2) # 2 * sigma space squared
    sigma =  0.3*((k_size-1)*0.5 - 1) + 0.8
    gaus_kernel = gaussianKernel(k_size,sigma,True)
    ky,kx = np.int32(np.floor(gaus_kernel.shape[0]/2)), np.int32(np.floor(gaus_kernel.shape[1]/2)) if gaus_kernel.shape[1] != 1 else 1
    padded_img = getPaddedArray(in_image, gaus_kernel)
    x_matrix = np.repeat(np.arange(padded_img.shape[1]).reshape(1,-1), padded_img.shape[0],axis=0) # repeat x indexes rows times
    y_matrix = np.repeat(np.arange(padded_img.shape[0]).reshape(1,-1), padded_img.shape[1],axis=0).transpose() # repeat y indexes cols times
    for i in range(ky, in_image.shape[0] + ky):
        for j in range(kx, in_image.shape[1] + kx):
            pivoti = in_image[i-ky,j-kx]
            neighborsi = padded_img[
                i - ky : i + ky,
                j - kx : j + kx
                                    ]
            intensity_diff = np.abs(pivoti - neighborsi)
            i_diff_gaussian = np.e ** ((-1* intensity_diff) / sig_s_sq)
            neighborsx = x_matrix[i - ky : i + ky, j - kx : j + kx] # get the x values of the neighbors
            neighborsy = y_matrix[i - ky : i + ky, j - kx : j + kx] # get the y values of the neighbors
            p_norm = (neighborsx - j) **2 + (neighborsy - i) **2 # calc norm based on indices
            p_norm_gaussian = np.e ** (-1* p_norm / sig_c_sq)
            deno = np.sum(p_norm_gaussian * i_diff_gaussian)
            nomi = np.sum(i_diff_gaussian * p_norm_gaussian * neighborsi)
            result[i-ky,j-kx] = nomi / deno
    return bil_cv, result

def bilateral_filter(image, d, sigma_color, sigma_space):
    # Create a padded image with reflection border
    padded_image = cv2.copyMakeBorder(image, d, d, d, d, cv2.BORDER_REFLECT)

    # Create the spatial distance kernel
    x, y = np.meshgrid(np.arange(-d, d + 1), np.arange(-d, d + 1))
    kernel_space = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    # Initialize the filtered image
    filtered_image = np.zeros_like(image)

    for channel in range(image.shape[2]):
        # Extract the channel
        channel_image = padded_image[:, :, channel]

        # Create the intensity difference kernel
        kernel_color = np.exp(-((channel_image - channel_image[d:-d, d:-d])**2) / (2 * sigma_color**2))

        # Convolve the image with the separable kernels
        filtered_channel = cv2.filter2D(channel_image, -1, kernel_space, borderType=cv2.BORDER_CONSTANT)
        filtered_channel = cv2.filter2D(filtered_channel, -1, kernel_color, borderType=cv2.BORDER_CONSTANT)

        # Store the filtered channel in the filtered image
        filtered_image[:, :, channel] = filtered_channel[d:-d, d:-d]

    return filtered_image


