import numpy as np
import cv2 as cv
import pywt
import math
import imutils

def rotation(image):
    image = imutils.rotate(image,10)
    return image

def avgfilter(img):
    m,n = img.shape

    mask = np.ones([3,3],dtype=int)
    mask = mask/9

    newimg = np.zeros([m,n])

    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

            newimg[i,j] = temp

    # newimg = newimg.astype(np.uint8)
    return newimg 

def mult(mat1,mat2):
    ans = 0
    n = len(mat1)
    m = len(mat2)
    for i in range(n):
        for j in range(m):
            ans += mat1[i][j] * mat2[i][j]
    return ans

def ncc(mat1,mat2):
    mat1 = np.array(mat1, dtype=np.float64)
    mat2 = np.array(mat2, dtype=np.float64)
    num1 = mult(mat1,mat2)
    num2 = mult(mat1,mat1)
    num3 = mult(mat2,mat2)
    num4 = (math.sqrt(num2*num3))
    return num1/num4

def psnr(mat1,mat2):

    ans = 0
    n = len(mat1)
    m = len(mat2)
    for i in range(n):
        for j in range(m):
            ans += (mat1[i][j] - mat2[i][j])**2
    mse = ans/(n*m)

    num = (255**2)/mse
    res = 10 * math.log(num,10)
    return res

def dwt(coverImg,watermarkImg):
    cv.waitKey(0)
    coverImg = cv.resize(coverImg,(512,512))
    cv.imshow("cover",coverImg)
    watermarkImg = cv.resize(watermarkImg,(256,256))
    cv.imshow("logo",watermarkImg)

    coverImg = np.float64(coverImg)
    # print(coverImg)
    coverImg /= 255
    # print(coverImg)
    coeffc = pywt.dwt2(coverImg,"haar")
    cA , (cH , cV , cD) = coeffc
    # print(coeff)
    # print(cA , cH, cV , cD)
    
    watermarkImg = np.float64(watermarkImg)
    watermarkImg /= 255

    a_c = 1
    a_w = 0.1
    coeffw = (a_c * cA + a_w * watermarkImg, (cH ,cV ,cD))
    watermarkedImg = pywt.idwt2(coeffw,"haar")

    # 
    # Attacks
    # 

    # watermarkedImg = cv.rotate(watermarkedImg,cv.ROTATE_180)
    # watermarkedImg = rotation(watermarkedImg)
    # watermarkedImg = avgfilter(watermarkedImg)

    cv.imshow("watermarkedImg",watermarkedImg)
    psnrval = psnr(coverImg,watermarkedImg)
    print(psnrval)

    coeffWM = pywt.dwt2(watermarkedImg,"haar")
    hA , (hH , hV , hD) = coeffWM

    extracted = (hA-a_c*cA)/a_w
    extracted *= 255
    extracted = np.uint8(extracted)

    cv.imshow("Extracted watermarkImg",extracted)

    nccval = ncc(watermarkImg,extracted)
    print(nccval)
    cv.imwrite("watermarkedImg.png",watermarkedImg*255)
    cv.imwrite("extracted.png",extracted)


    cv.waitKey(5000)
    cv.destroyAllWindows()


coverImg = cv.imread("images/PA-2.tif",0)
watermarkImg = cv.imread("images/mnnit-logo.jpeg",0)

dwt(coverImg,watermarkImg)