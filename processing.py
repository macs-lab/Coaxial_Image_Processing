import cv2

import numpy as np
import matplotlib.pyplot as plt

'''
image preprocessing
adjust lightness and contrast
reference : https://blog.csdn.net/qq_40755643/article/details/84032773
'''
def zeros_like(img):
    # 直方图归一化
    dst1 = np.zeros_like(img)
    cv2.normalize(img, dst1, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #公式
    # cv2.imwrite("data/zeros_like.png",dst1)
    return dst1
    
def equalizeHist(img):
    #直方图均衡化
    dst2 = cv2.equalizeHist(img)    
    # cv2.imwrite("data/equalizeHist.png",dst2)
    return dst2
    
def CLAHE(img):
    # 限制对比度自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    dst3 = clahe.apply(img)
    # cv2.imwrite("data/CLAHE.png",dst3)
    return dst3

# mul means multiple
def line_trans(img,mul = 1.2):
    mul = 0.8
    O = mul * img
    O[O>255] = 255
    O = np.round(O)
    O = O.astype(np.uint8)
    # cv2.imwrite("data/linear_trans.png",O)
    return O

def gama(img,gama = 0.9):
    # gama 变换
    gamma = 0.9
    O = np.power(img, gamma)
    # cv2.imwrite("data/gama_trans.png",O)
    return O
    
if __name__ == "__main__":

    src = cv2.imread('data/404.jpeg', cv2.IMREAD_ANYCOLOR)
    method = {1:zeros_like, 2:equalizeHist, 3:CLAHE, 4:line_trans, 5:gama}
    methodName = {1:'zeros_like', 2:'equalizeHist', 3:'CLAHE', 4:'line_trans', 5:'gama'}
    
    # resize the original image
    scale_percent = 20 # percent of original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
    # cv2.imwrite("data/390/prep/404_resized.png",resized)
    
    # gaussian = cv2.GaussianBlur(src, (5, 5), 1)
    # cv2.imwrite("data/390/prep/404_gaussian.png",gaussian)
    src = cv2.imread("data/390/prep/404_zeros_like.png", cv2.IMREAD_ANYCOLOR)
    # for i in range(1,5):
    #     k = 5*i
    #     if(k%2 == 0 ):
    #         k = k+1
    #     median = cv2.medianBlur(src, k)
    #     cv2.imwrite("data/390/prep/404_zeros_median"+str(k)+".png",median)
    
    # blur = cv2.bilateralFilter(src, 9, 75, 75)
    # cv2.imwrite("data/390/prep/404_bilateral.png",blur)
    # for i in range(1,10):
    #     kernel = np.ones((5*i, 5*i), np.uint8)
    #     # erosion = cv2.erode(src, kernel)  
    #     # cv2.imwrite("data/390/prep/404_erosion.png",erosion)
        
    #     dilation = cv2.dilate(src, kernel)
    #     cv2.imwrite("data/390/prep/404_dilation_"+str(5*i)+".png",dilation)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # for i in range(1,10):
    #     k = 3+i*3
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    #     opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    #     cv2.imwrite("data/390/prep/404_opening_"+str(k)+".png",opening)
    # # resize the prep img
    
    # for key in range(1,6):
    #     img = method[key](src)
        
    #     # resize image
    #     resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #     src = resized
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+".png",resized)
        
        
    #     gaussian = cv2.GaussianBlur(src, (5, 5), 1)
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"gaussian.png",gaussian)
        
    #     median = cv2.medianBlur(src, 5)
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"median.png",median)
        
    #     blur = cv2.bilateralFilter(src, 9, 75, 75)
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"bilateral.png",blur)
        
    #     kernel = np.ones((10, 10), np.uint8)
    #     erosion = cv2.erode(src, kernel)  
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"erosion.png",erosion)
        
    #     dilation = cv2.dilate(src, kernel)
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"dilation.png",dilation)
        
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    #     cv2.imwrite("data/390/prep/404_"+str(methodName[key])+"opening.png",opening)
 

    # cv2.imshow("img", src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


