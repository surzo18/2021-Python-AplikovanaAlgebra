import cv2
from Convolution import gaussian_blur_kernel
import numpy as np
import matplotlib.pyplot as plt

def applicateBlurOnPaddedImage( width, height, blur_kernel):
    blured_matrix = np.zeros(shape=(width*height,height*width))

    for key in range(width*height):

        isFirstLine = False
        isLastLine = False
        isFirstColumn =False
        isLastColumn = False

        #Solve corners first
        if(key == 0):
             blured_matrix[key][width * height-1] = blur_kernel[0][0] #top-left-corner

        if(key == width-1):
            blured_matrix[key][width * height - width ] = blur_kernel[0][2]  # top-right-corner

        if(key == width * height - width):
            blured_matrix[key][width-1] = blur_kernel[2][0] #bottom-left-corner

        if(key == width * height - 1):
            blured_matrix[key][0] = blur_kernel[2][2] #bottom-right-corner

        #Solve first_line
        if(key < width):
            isFirstLine = True
            if(key !=0): #corner
                blured_matrix[key][width * height - width -1 + key ] = blur_kernel[0][0]

            blured_matrix[key][width * height - width  + key] = blur_kernel[0][1]

            if(key !=width-1): #corner
                blured_matrix[key][width * height - width + key+1] = blur_kernel[0][2]


        #Solve last-line
        if (key >= width*height - width):
            isLastLine = True
            if (key != width*height - width):  # corner
                blured_matrix[key][width - (width * height - key) - 1] = blur_kernel[2][0]

            blured_matrix[key][width - (width * height - key )] = blur_kernel[2][1]

            if (key != width*height - 1):  # corner
                blured_matrix[key][ (key % width) + 1 ] = blur_kernel[2][2]
                # 3 -2 = 1
                # 3 - 3 = 2
        #Solve first-column
        if(key % width == 0):
            isFirstColumn = True
            if (key != 0):  # corner
                blured_matrix[key][key-1] = blur_kernel[0][0]

            blured_matrix[key][key - 1 + width] = blur_kernel[1][0]

            if (key != width * height -width):  # corner
                blured_matrix[key][key - 1 + 2*width] = blur_kernel[2][0]

        #Solve-last-column
        if(key % width == width-1):
            isLastColumn = True
            if (key != width - 1):  # corner
                blured_matrix[key][key - 2*width+1 ] = blur_kernel[0][2]

            blured_matrix[key][key - width+1] = blur_kernel[1][2]

            if (key != width * height - 1):  # corner
                blured_matrix[key][key + 1] = blur_kernel[2][2]

        #SolveOthers
        if(isFirstLine == False and isFirstColumn == False):
            blured_matrix[key][key - width - 1] = blur_kernel[0][0]

        if(isFirstLine == False):
            blured_matrix[key][key - width] = blur_kernel[0][1]

        if(isFirstLine == False and isLastColumn == False):
            blured_matrix[key][key - width + 1] = blur_kernel[0][2]


        #--------------------
        if(isFirstColumn == False):
            blured_matrix[key][key-1] = blur_kernel[1][0]

        #Center
        blured_matrix[key][key] = blur_kernel[1][1]

        if(isLastColumn == False):
            blured_matrix[key][key+1] = blur_kernel[1][2]

        #--------------
        if (isLastLine == False and isFirstColumn == False):
            blured_matrix[key][key + width - 1] = blur_kernel[2][0]

        if (isLastLine == False):
            blured_matrix[key][key + width] = blur_kernel[2][1]

        if (isLastLine == False and isLastColumn == False):
            blured_matrix[key][key + width + 1] = blur_kernel[2][2]

    return blured_matrix


origin_image_matrix = cv2.imread('Assets/plus.png')
height, width, channels = origin_image_matrix.shape
plt.imshow(origin_image_matrix, interpolation='none', cmap='gray')
plt.title("Origin-Image")
plt.show()
print("width: " + str(width) + " height: " + str(height))
blur_kernel_matrix = gaussian_blur_kernel(3, verbose=True)
#blur_kernel_matrix = [[1,2,3],[4,5,6],[7,8,9]]

#Testing matrix
#matrix = [[i for i in range(3)] for i in range(3)]

#MATRIX B
B = applicateBlurOnPaddedImage(width=width,height=height,blur_kernel=blur_kernel_matrix)
#B = applicateBlurOnPaddedImage(width,height,blur_kernel_matrix)

# MATRIX A -  origin image flatten
A = np.zeros(( width*height, 1))
flatten_origin = origin_image_matrix.transpose().flatten()
for i in range(width*height):
    A[i][0] = flatten_origin[i]

# A * B => SHOULD BE OUTPUT (Blured  - O)
Output = np.dot(B, A)
Output_reshaped = Output.reshape(width,height)

plt.imshow(Output_reshaped, cmap='gray')
plt.title("Blured - Output")
plt.show()

# OUTPUT * B+ => SHOULD BE APPROX. INPUT
U, D, V = np.linalg.svd(B)
D_plus = np.zeros((B.shape[0], B.shape[1])).T
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
B_plus = V.T.dot(D_plus).dot(U.T)

repaired_image = np.dot(B_plus,Output)
repaired_image = repaired_image.reshape(width,height)
plt.imshow(repaired_image, interpolation='none', cmap='gray')
plt.title("Repaired")
plt.show()




