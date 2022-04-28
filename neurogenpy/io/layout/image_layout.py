"""
Image layout module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import math
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from skimage.measure import find_contours

from .graph_layout import GraphLayout


class ImageLayout(GraphLayout):
    """
    Image layout class.
    """

    def __init__(self, graph, image_url, threshold=200):
        """
        Image layout class constructor.

        Parameters
        ----------
        graph : networkx.DiGraph
            Graph whose layout has to be computed.

        image_url : str
            URL of the image used to calculate the layout.

        threshold : int, default=200
            Threshold used to binarize the image pixels.
        """

        super().__init__(graph)
        response = requests.get(image_url)
        self.img = Image.open(BytesIO(response.content))
        self.threshold = threshold

    def run(self, env='neurogenpy'):
        """
        Calculates the layout for the graph with the image algorithm.

        Parameters
        ----------
        env : str, default='neurogenpy'
            Environment used to calculate the layout.

        Returns
        -------
        dict
            A dictionary with the nodes IDs as keys and their coordinates as
            values.

        Raises
        ------
        ValueError
            If the environment selected is not supported.
        """

        if env == 'neurogenpy':
            return self._run_neurogenpy()
        else:
            raise ValueError(f'{env} environment is not supported.')

    def _run_neurogenpy(self):
        nodes = list(self.graph.nodes())

        pixels = self._monochrome()
        contours = find_contours(pixels, 0.8)
        if not contours:
            raise Exception(
                'Edges couldn\'t be detected. Try changing the threshold')
        x = [elem for contour in contours for elem in contour[:, 1]]
        y = [elem for contour in contours for elem in contour[:, 0]]

        len_x = len(x)
        factor = math.ceil(len(x) / len(nodes)) if len(nodes) < len_x else 1

        return {node: (float(x[(i * factor < len_x) * i * factor]),
                       float(y[(i * factor < len_x) * i * factor])) for i, node
                in enumerate(nodes)}

    def _monochrome(self):
        """Converts the image to monochrome."""

        image = self.img.convert('L')
        image_array = np.array(image)

        # Binarize a numpy array using threshold as cutoff
        image_array = (image_array > self.threshold) * 255
        image = Image.fromarray(image_array)

        # convert image to black and white
        bw_img = image.convert(mode='1', dither=2)
        return 1 - np.asarray(bw_img).astype(int)
