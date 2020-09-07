import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

from src import io_utils

class Tile:
    """
    Class for creating and manipulating a single tile.
    Tiles are assumed to be square.

    Attributes:
        img: image, assumed to have 1 tile in it
    """

    def __init__(self, img):
        if img.shape[0] != img.shape[1]:
            raise ValueError('Image must be square to be a tile (image shape is {}x{}).'.format(*img.shape))
        self.img = img
        self.dims = img.shape

    # Plotting
    @classmethod
    def plot(cls, tile):
        """
        Plots a tile

        :param tile: tile to plot
        """
        plt.imshow(tile.img)
        plt.axis('off')
        plt.show()

    @classmethod
    def plot_grid(cls, tile_list, rows=None, cols=None):
        """
        Plots tiles in tile_list in a grid of rows x cols.

        If none rows and cols are None, plots every tile in tile_list in
        square grid of ceil(len(tile_list) ** 0.5).

        If one of rows and cols are specified, figures out the other dim as
        ceil(len(tile_list) / <given dimension>).

        If rows * cols > len(tile_list), add empty cells to the grid.

        If rows * cols < len(tile_list), samples rows * cols elements from tile_list.

        :param tile_list: list of tiles to plot
        :param rows: grid dimension: rows
        :param cols: grip dimension: cols
        """
        io_utils.plot_sample_imgs([tile.img for tile in tile_list], rows=rows, cols=cols)
        
    # Flipping
    def flip_vertical(self):
        return Tile(cv2.flip(self.img, 0))

    def flip_horizontal(self):
        return Tile(cv2.flip(self.img, 1))

    def flip_transpose(self):
        return Tile(cv2.transpose(self.img))

    def rotate(self, clockwise=True):
        if clockwise:
            return self.flip_transpose().flip_horizontal()
        else:
            return self.flip_transpose().flip_vertical()

    # Cutting
    def get_quadrant(self, row, col):
        """
        Cuts out a quadrant from itself.

        row=0, col=0: upper left
        row=0, col=1: upper right
        row-1, col=0: lower left
        row-1, col=0: lower right

        :param row: index of quadrant to cut
        :param col: index of quadrant to cut
        :return: new tile cut out from current tile
        """
        if not (row in (0, 1) and col in (0, 1)):
            raise Exception('Row and col must be either 0 or 1')

        row_start = self.dims[0] // 2 * row
        row_end = row_start + self.dims[0] // 2

        col_start = self.dims[1] // 2 * col
        col_end = col_start + self.dims[1] // 2

        return Tile(self.img[row_start:row_end, col_start:col_end])
    
    def get_pieces(self, n = 9):
        """
        Returns n pieces; original image is cut in nxn by row and col

        :param n: number of pieces to cut by row and col
        :return: nxn new tiles cut out from current tile
        """
        
        tile_list = []
        
        for i in range(n):
            for j in range(n):
                row_start = self.dims[0] // n * i
                row_end = self.dims[0] // n * (i+1)

                col_start = self.dims[1] // n * j
                col_end = self.dims[1] // n * (j+1)
                
                tile_list.append(Tile(self.img[row_start:row_end, col_start:col_end]))

        return tile_list
        
    def get_square_from_center(self, ratio=0.8):
        """
        Cuts sub-tile from center of itself.
        Size of result is equal ratio * size of self.

        :param ratio: relative size of new tile
        :return: new tile cut out from current tile
        """
        if ratio >= 1 or ratio <= 0:
            raise Exception('Ratop must be between 0 and 1')

        row_start = int(self.dims[0] * (1 - ratio) / 2.0)
        row_end = int(self.dims[0] * (1 + ratio) / 2.0)

        col_start = int(self.dims[1] * (1 - ratio) / 2.0)
        col_end = int(self.dims[1] * (1 + ratio) / 2.0)

        return Tile(self.img[row_start:row_end, col_start:col_end])

    def get_rhombus(self):
        """
        Cuts out a rhombus from self, where vertices of rhombus are centers of edges of self.

        :return: new tile cut out from current tile
        """
        # Cutting points
        cut_points = np.array([
            [0, self.dims[1] // 2],
            [self.dims[0] // 2, 0],
            [self.dims[0], self.dims[1] // 2],
            [self.dims[0] // 2, self.dims[1]]
        ])

        rect = cv2.minAreaRect(cut_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # Straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(self.img, M, (width, height))
        return Tile(warped)
    
    def remove_center(self):
        """
        Assembles new tile by cutting tile into 9 equal pieces
        and then assembling a new one from corners:
        A|B|C
        D|E|F => A|C
        G|H|I    G|I
        """
        subtiles = self.get_pieces(3)

        result = np.concatenate(
            (
                np.concatenate(
                    (
                        subtiles[0].img,
                        subtiles[2].img
                    ),
                    axis=0
                ),
                np.concatenate(
                    (
                        subtiles[6].img,
                        subtiles[8].img
                    ),
                    axis=0
                )
            ),
            axis=1
        )

        return Tile(result)

    # Assembling
    def assemble_quadrant_unfold(self, row, col):
        """
        Assembles new tile by mirrowing and attaching self, starting from position row, col:

        AB    AB | BA
        CD => CD | DC
              -------
              CD | DC
              AB | BA

        row=0, col=0: upper left
        row-0, col=1: upper right
        row=1, col=0: lower left
        row=1, col=1: lower right

        :param row: position to start assembling from
        :param col: position to start assembling from
        :return: new tile glued from current tile
        """
        if not (row in (0, 1) and col in (0, 1)):
            raise Exception('Row and col must be either 0 or 1')

        seed_img = self.img

        if row == 1:
            seed_img = cv2.flip(self.img, 0)

        if col == 1:
            seed_img = cv2.flip(self.img, 1)

        result = np.concatenate(
            (
                np.concatenate(
                    (
                        seed_img,
                        cv2.flip(seed_img, 0)
                    ),
                    axis=0
                ),
                np.concatenate(
                    (
                        cv2.flip(seed_img, 1),
                        cv2.flip(cv2.flip(seed_img, 0), 1)
                    ),
                    axis=0
                )
            ),
            axis=1
        )

        return Tile(result)

    def assemble_quadrant_windmill(self, clockwise=True):
        """
        Assembles new tile by rotating and attaching self, clockwise or counter-clockwise.

        AB    AB | CA
        CD => CD | DB
              -------
              BD | DC
              AC | BA

        :param clockwise: direction
        :return: new tile glued from current tile
        """
        new_size = min(self.dims[0], self.dims[1])
        new_img = self.img[self.dims[0] - new_size:self.dims[0], self.dims[1] - new_size:self.dims[1]]

        upper_left = new_img
        upper_right = Tile(upper_left).rotate(clockwise=clockwise).img
        lower_right = Tile(upper_right).rotate(clockwise=clockwise).img
        lower_left = Tile(lower_right).rotate(clockwise=clockwise).img

        img_upper = np.concatenate((upper_left, upper_right), axis=1)
        img_lower = np.concatenate((lower_left, lower_right), axis=1)

        img_full = np.concatenate((img_upper, img_lower), axis=0)
        return Tile(img_full)

    # Checks
    # To do: symmetry checks
