from .BnLayout import BnLayout

import math
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from skimage import measure


class LayoutImage(BnLayout):
    """
    Image Layout class.
    """

    def __init__(self, graph, graph_initial_width=None,
                 graph_initial_height=None, additional_params=None):
        """
        Image Layout class constructor.
        @param graph:
        @param graph_initial_width:
        @param graph_initial_height:
        @param additional_params:
        """
        super(LayoutImage, self).__init__(graph, graph_initial_width,
                                          graph_initial_height)
        if additional_params is not None:
            self.image_url = additional_params["layout-image-url"]
            self.threshold_edges = int(
                additional_params["layout-image-threshold"])

    def run(self):
        nodes = list(self.graph.nodes())

        self.layout = {}

        """
        Concha: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlBQhYP-U_sCs52e4OXeaXjBffs9tXITn58hjnODZp2YmkpqOebA
        Pedro: http://dia.fi.upm.es/sites/default/files/styles/medium/public/Pedro.jpg
        CIG: http://cig.fi.upm.es/sites/default/files/logo_CIG.png
        Pikachu: https://i.pinimg.com/originals/a6/30/cc/a630cceb36c8d3eab892c9d8442ecef1.png
        Human: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0_-UuZrU4jOSriI2xSTvlIqWjaPo3ClHU10EwX5UKAMAtg8F6
        Cat: https://i.ebayimg.com/images/g/Y2AAAOSwdjNZCL6D/s-l300.jpg
        ADN: https://assetsnffrgf-a.akamaihd.net/assets/m/102015282/univ/art/102015282_univ_cnt_2_xl.jpg
        """

        image = ImageObject(self.image_url)
        image.monochrome(scale=2, threshold=self.threshold_edges)
        contours = measure.find_contours(image.pixels, 0.8)
        if len(contours) == 0:
            raise Exception(
                "Edges couldn't be detected. Try changing the threshold")
        x = []
        y = []
        for n, contour in enumerate(contours):
            x += contour[:, 1].tolist()
            y += contour[:, 0].tolist()

        factor = 1
        if len(nodes) < len(x):
            factor = math.ceil(len(x) / len(nodes))
        for i, node in enumerate(nodes):
            if i * factor < len(x):
                self.layout[node] = (
                    float(x[i * factor]), float(y[i * factor]))
            else:
                self.layout[node] = (float(x[0]), float(y[0]))

        return self.layout


class ImageObject:
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.img = Image.open(BytesIO(response.content))
        self.og_size = self.img.size

    def monochrome(self, scale=2, threshold=200):

        # convert image to monochrome
        image = self.img.convert('L')
        image_array = np.array(image)

        # Binarize a numpy array using threshold as cutoff
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if image_array[i][j] > threshold:
                    image_array[i][j] = 255
                else:
                    image_array[i][j] = 0

        image = Image.fromarray(image_array)

        # scale image down to reduce number of non-zero pixels
        # img_sm = image.resize(tuple([int(v / scale) for v in image.size]), Image.ANTIALIAS)

        # convert image to black and white
        img_bw = image.convert(mode='1', dither=2)
        self.bw_img = img_bw
        self.pixels = (1 - np.asarray(img_bw).astype(int))
        self.pixels_flat = np.reshape(self.pixels, self.pixels.size)
