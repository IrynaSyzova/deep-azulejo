import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class Tile:
    """
    Class for creating and manipulating a single tile.
    Tiles are assumed to be square.

    Attributes:
        img: image, assumed to have 1 tile in it
    """

    def __init__(self, img):
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
        if (rows is None) or (rows == 0) or (rows > len(tile_list)):
            if (cols is None) or (cols == 0):
                rows = int(math.ceil(len(tile_list) ** 0.5))
                cols = int(math.ceil(len(tile_list) / rows))
            else:
                rows = int(math.ceil(len(tile_list) / cols))
        elif (cols is None) or (cols == 0) or (rows > len(tile_list)):
            cols = int(math.ceil(len(tile_list) / rows))

        if (rows == 0) or (cols == 0):
            print('Nothing to plot')
            return

        if rows * cols >= len(tile_list):
            tile_list_plot = np.array(tile_list + [np.nan] * (rows * cols - len(tile_list))).reshape(rows, cols)

        else:
            tile_list_plot = np.random.choice(tile_list, size=(rows, cols), replace=False)

        fig, ax = plt.subplots(rows, cols, figsize=(16, (16 / cols) * rows))
        # 16 is the right width for my screen
        # height is calculated to keep same distance horizontally and vertically between plotted tiles

        # if rows ==1 or cols == 1 then plr.subplots(rows, cols) will create a vector of axis
        # otherwise subplots(roxws, cols) will create a matrix of axis
        if rows == 1:
            for col in range(cols):
                ax[col].imshow(tile_list[col].img)
                ax[col].axis('off')
        elif cols == 1:
            for row in range(rows):
                ax[row].imshow(tile_list[row].img)
                ax[row].axis('off')
        else:
            for row in range(rows):
                for col in range(cols):
                    if tile_list_plot[row][col] is not np.nan:
                        ax[row, col].imshow(tile_list_plot[row][col].img)
                        ax[row, col].axis('off')
                    else:
                        ax[row, col].axis('off')

        plt.show()

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