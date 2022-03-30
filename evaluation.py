import os
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\OCR\tesseract.exe'
path_gt = r'E:\doc\DocUNet\GT_img_bench'

def Levenshtein_Distance(str1, str2):
    """
        - param str1
        - param str2
        - return
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


def evaluate(path_ours, tail):
    print(path_ours)
    N = 66
    cer1 = []
    cer2 = []
    ed1 = []
    ed2 = []
    lis = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 19, 20, 21, 22, 23, 24, 27, 30, 31, 
           32, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    for i in range(1, N):
        if i in lis:
            # print(i)
            gt = Image.open(path_gt + '\\'+ str(i) + '.png')
            img1 = Image.open(path_ours + '\\' + str(i) + '_1' +  ' copy' + tail + '.png')
            img2 = Image.open(path_ours + '\\' + str(i) + '_2' +  ' copy' + tail + '.png')

            content_gt = pytesseract.image_to_string(gt)
            content1 = pytesseract.image_to_string(img1)
            content2 = pytesseract.image_to_string(img2)
            l1 = Levenshtein_Distance(content_gt, content1)
            l2 = Levenshtein_Distance(content_gt, content2)
            ed1.append(l1)
            ed2.append(l2)
            cer1.append(l1 / len(content_gt))
            cer2.append(l2 / len(content_gt))
    print('CER: ', np.mean(cer1+cer2))
    print('ED:  ', np.mean(ed1 + ed2))


path = r'F:\ablation\conv'
for i in os.listdir(path):
    path_dir = os.path.join(path, i)
    evaluate(path_ours=path_dir, tail='_rec')