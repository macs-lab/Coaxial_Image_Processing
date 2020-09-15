import argparse
import logging
import time
from graph import build_graph, segment_graph
from random import random
from PIL import Image, ImageFilter,ImageFont, ImageDraw
from skimage import io
import numpy as np
import copy

import cv2

dict_color = {}
def diff(img, x1, y1, x2, y2):
    # _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    # return np.sqrt(_out)
    return np.linalg.norm(img[x1, y1] - img[x2, y2])



def threshold(size, const):
    return (const * 1.0 / size)

'''
draw text reference:https://stackoverflow.com/questions/58968752/loading-fonts-in-python-pillow-on-a-mac
'''
def generate_image(forest, width, height,thres):
    global dict_color

    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in range(width*height)]
    # dict_color = {}
    img = Image.new('RGB', (width, height))
    im = img.load()
    # im = ImageDraw.Draw(im)
    # font = ImageFont.truetype("Keyboard.ttf",16)
    font = ImageFont.truetype("Keyboard.ttf",16)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]
            if comp not in dict_color:
                dict_color[comp] = [x,y,colors[comp]]
                # print("camp and color ",str(comp),str(colors[comp]))
                # cv2.circle(im,(x,y),3,(0,0,255),-1)
                print(comp,thres[comp])
                # text = str(comp) + ' ' + str(thres[comp])
                # text = "hello"
                # draw.text((x, y), text, (255, 255, 255), font=font)

                # cv2.putText(im, str(comp), (x,y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    # print(dict_color)
    # draw = ImageDraw.Draw(img)
    # for i in dict_color:
    #     loc = dict_color[i]
    #     num = int(thres[i]*100)
    #     draw.text(loc,str(num ),(255,255,255),font=font)
    # img.save("./data/390/test/seg/_395_label.png")
    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


def get_segmented_image(sigma, neighbor, K, min_comp_size, input_file, output_file,logger):
    if neighbor != 4 and neighbor!= 8:
        logger.warn('Invalid neighborhood choosed. The acceptable values are 4 or 8.')
        logger.warn('Segmenting with 4-neighborhood...')
    start_time = time.time()
    image_file = Image.open(input_file)

    size = image_file.size  # (width, height) in Pillow/PIL
    logger.info('Image info: {} | {} | {}'.format(image_file.format, size, image_file.mode))

    # Gaussian Filter
    # smooth = image_file.filter(ImageFilter.GaussianBlur(sigma))
    # smooth = np.array(smooth)
    
    smooth = np.array(image_file)
    logger.info("Creating graph...")
    graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor==8)
    
    logger.info("Merging graph...")
    forest, thres = segment_graph(graph_edges, size[0]*size[1], K, min_comp_size, threshold)

    logger.info("Visualizing segmentation and saving into: {}".format(output_file))
    image = generate_image(forest, size[1], size[0],thres)



    image.save(output_file)
    label(image, thres,output_file)

    logger.info('Number of components: {}'.format(forest.num_sets))
    logger.info('Total running time: {:0.4}s'.format(time.time() - start_time))
    return image

def label(image,thres,out):
    global dict_color
    # img = img.load()
    print("label")
    img = image
    font = ImageFont.truetype("Keyboard.ttf",16)
    draw = ImageDraw.Draw(img)
    for i in dict_color:
        x = dict_color[i][1]
        y = dict_color[i][0]
        loc = dict_color[i]
        num = int(thres[i]*100)
        te = str(num)
        draw.text((x,y),te,(255,255,255),font=font)
        a, b, c = dict_color[i][2]
        lab = str(a) + ' ' + str(b) + ' ' + str(c) + '\n ' + te
        print(lab)
    name = "."+out.split('.')[1] + "_label.png"
    img.save(name)
    # draw.fill((100,100,100))
    # draw.rectangle((0,0,260,200), fill=(0, 0, 0, 0))
    # img.save("./data/390/test/seg/_395_label.png")
    # draw2 = ImageDraw.Draw(image)
    # for i in dict_color:
    #     x = dict_color[i][1]
    #     y = dict_color[i][0]
    #     a,b,c = dict_color[i][2]
    #     text = str(a)+' '+str(b)+' '+str(c)
    #     draw2.text((x,y),text,(255,255,255),font=font)
    # name = "." + out.split('.')[1] + "_lr.png"
    # image.save(name)
    # return img
    
    
if __name__ == '__main__':
    # argument parser
    # parser = argparse.ArgumentParser(description='Graph-based Segmentation')
    # parser.add_argument('--sigma', type=float, default=1.0, 
    #                     help='a float for the Gaussin Filter')
    # parser.add_argument('--neighbor', type=int, default=8, choices=[4, 8],
    #                     help='choose the neighborhood format, 4 or 8')
    # parser.add_argument('--K', type=float, default=10.0, 
    #                     help='a constant to control the threshold function of the predicate')
    # parser.add_argument('--min-comp-size', type=int, default=2000, 
    #                     help='a constant to remove all the components with fewer number of pixels')
    # parser.add_argument('--input-file', type=str, default="./assets/seg_test.jpg", 
    #                     help='the file path of the input image')
    # parser.add_argument('--output-file', type=str, default="./assets/seg_test_out.jpg", 
    #                     help='the file path of the output image')
    
    # img = cv2.imread('./data/preprocessed/zeros_like.png')
    # resized = cv2.resize(img, None, fx=0.2, fy=0.2)
    # cv2.imwrite('./data/preprocessed/zeros_like_resized.png', resized)
    
    methodName = {1:'zeros_like', 2:'equalizeHist', 3:'CLAHE', 4:'line_trans', 5:'gama'}
    i = 4
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
    logger = logging.getLogger(__name__)
    sigma = 0.8
    neighbor = 8
    K = 10
    min_comp_size = 3000

    # equalizeHist
    # i, sigma, K = 2, 0.4,200
    # i, sigma, K = 2, 0.5, 9

    # zeros_like
    # i, sigma, K = 1,0.5,9

    # line_trans
    # i , sigma , K = 4, 0.6, 16

    # CLAHE
    # i , sigma , K = 3, 0.7, 10

    # gamma
    # i, sigma, K = 5,0.5,8

    # sigma,K = 0.7,12
    # dict_color = {}
    # text = "_s" + str(sigma) + "_K" + str(K) + "_minC" + str(min_comp_size)
    # # input_file = "./data/390/prep/680_" + str(methodName[i]) + ".png"
    # # output_file = "./data/390/segmentation/680_" + str(methodName[i]) + text+".png"
    input_file = "./data/390/prep/680_croppedCLAHE.png"
    output_file = "./data/390/segmentation/680_cropped_CLAHE.png"
    img = get_segmented_image(sigma, neighbor, K, min_comp_size, input_file, output_file,logger)
    labeled = label(img)
    # labeled.save("./data/390/segmentation/680_" + str(methodName[i]) +text+ "_label.png")
    labeled.save("./data/390/segmentation/680_cropped_CLAHE_label.png")
    print(dict_color)

    # for s in range(6):

    #     for k in range(10):
    #         i = 2
    #         sigma = 0.4+ s*0.1
    #         min_comp_size = 2000
    #         # K = 8+2*k
    #         K = 100+30*k

    #         dict_color = {}
    #         text = "_s" + str(sigma) + "_K" + str(K) + "_minC" + str(min_comp_size)
    #         # input_file = "./data/390/prep/716_" + str(methodName[i]) + ".png"
    #         # output_file = "./data/390/segmentation/716_" + str(methodName[i]) + text+".png"
    #         input_file = "./data/390/prep/680_croppedCLAHE.png"
    #         output_file = "./data/390/segmentation/680/C" + text+".png"
    #         img = get_segmented_image(sigma, neighbor, K, min_comp_size, input_file, output_file)
    #         # labeled = label(img)
    #         # labeled.save("./data/390/segmentation/680_C"+text+ "_label.png")
