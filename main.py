import colorsys
import math
import PySimpleGUI as sg
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# max_lim_xy = 1000
# center_xy = 500
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shape:
    def get_area(self):
        pass

    def draw_shape(self):
        pass

    @classmethod
    def get_shape_name(cls):
        pass

    @staticmethod
    def get_shape(name, input_values, error):
        try:
            if name == 'Circle' or name == 'Square' or name == 'Sphere' or name == 'Cube':
                x = input_values[0]
                if name == 'Circle':
                    return Circle(x)
                elif name == 'Square':
                    return Square(x)
                elif name == 'Sphere':
                    return Sphere(x)
                else:
                    return Cube(x)
            if shape_name == 'Rectangle' or shape_name == 'Triangle' or shape_name == 'Rhombus' \
                    or shape_name == 'Сylinder' or shape_name == 'Cone':
                x, y = input_values[0], input_values[1]
                if shape_name == 'Rectangle':
                    return Rectangle(x, y)
                elif shape_name == 'Triangle':
                    return Triangle(x, y)
                elif shape_name == 'Rhombus':
                    return Rhombus(x, y)
                elif shape_name == 'Сylinder':
                    return Cylinder(x, y)
                elif shape_name == 'Cone':
                    return Cone(x, y)
            if shape_name == 'Trapezoid' or shape_name == 'Parallelepiped' or shape_name == 'Pyramid':
                a, b, h = input_values[0], input_values[1], input_values[2]
                if shape_name == 'Trapezoid':
                    if a != b:
                        return Trapezoid(a, b, h)
                    else:
                        error.update(visible=True)
                elif shape_name == 'Parallelepiped':
                    return Parallelepiped(a, b, h)
                elif shape_name == 'Pyramid':
                    return Pyramid(a, b, h)

        except IndexError:
            error.update(visible=True)

    @staticmethod
    def get_shape_parameters(list_parameters, error):
        result = []
        try:
            for param in list_parameters:
                p = int(param)
                if p > 0:
                    result.append(int(param))
                else:
                    error.update(visible=True)
                    break
            return result
        except ValueError:
            error.update(visible=True)


class ThreeDShape:

    @classmethod
    def get_axes(cls, parameters):
        # plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('auto')
        return ax

    def get_volume(self):
        pass


class TwoDShape:

    @classmethod
    def get_axes(cls, parameters):
        max_parameter_value = 0
        for value in parameters:
            if value > max_parameter_value:
                max_parameter_value = value
        max_lim_xy = 1
        while max_lim_xy < max_parameter_value:
            max_lim_xy += 10
        max_lim_xy += 2 * max_lim_xy

        global center_xy
        center_xy = max_lim_xy // 2

        plt.xlim(0, int(max_lim_xy))
        plt.ylim(0, int(max_lim_xy))
        plt.grid()

        axes = plt.gca()
        axes.set_aspect('equal')
        return axes


class Circle(TwoDShape):

    @classmethod
    def get_shape_name(cls):
        return 'Circle'

    def __init__(self, r):
        super().__init__()
        self.__r = r

    def get_radius(self):
        return self.__r

    def set_radius(self, r):
        self.__r = r

    def get_area(self):
        return math.pi * self.__r ** 2

    def get_circumference(self):

        return 2 * math.pi * self.__r

    def draw_shape(self, axes):
        # fig = plt.figure()
        # fig.add_subplot()
        shape_patch = matplotlib.patches.Circle((center_xy, center_xy), self.__r, fill=True)
        axes.add_patch(shape_patch)
        return axes
        # plt.show()


class Square(TwoDShape):

    @classmethod
    def get_shape_name(cls):
        return 'Square'

    def __init__(self, x):
        super().__init__()
        self.__x = x

    def get_x(self):
        return self.__x

    def set_x(self, x):
        self.__x = x

    def get_area(self):
        return self.__x ** 2

    def get_perimeter(self):
        return 4 * self.__x

    def draw_shape(self, axes):
        shape_name = matplotlib.patches.Rectangle((0, 0), self.__x, self.__x, fill=True)
        axes.add_patch(shape_name)
        return axes
        # plt.show()


class Polygon(TwoDShape):

    def __get_points(self):
        pass

    def draw_shape(self, axes):
        shape_name = matplotlib.patches.Polygon(self.__get_points(), fill=True)
        axes.add_patch(shape_name)
        plt.show()


class Rectangle(TwoDShape):

    def __init__(self, x, y):
        super().__init__()
        self.__y = y
        self.__x = x

    @classmethod
    def get_shape_name(cls):
        return 'Rectangle'

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def set_x(self, x):
        self.__x = x

    def set_y(self, y):
        self.__y = y

    def get_area(self):
        return self.__x * self.__y

    def get_perimeter(self):
        return 2 * (self.__x + self.__y)

    def draw_shape(self, axes):
        shape_name = matplotlib.patches.Rectangle((0, 0), self.__x, self.__y, fill=True)
        axes.add_patch(shape_name)
        return axes
        # plt.show()


class Cube(ThreeDShape):

    def __init__(self, x):
        super().__init__()
        self.__x = x

    @classmethod
    def get_shape_name(cls):
        return 'Cube'

    def get_area(self):
        return 6 * self.__x ** 2

    def get_perimeter(self):
        return 12 * self.__x

    def draw_shape(self, ax):
        # Create axis
        sides = [self.__x, self.__x, self.__x]
        data = np.ones(sides, dtype=np.bool_)
        alpha = 0.9
        colors = np.empty(sides + [4], dtype=np.float32)
        colors[:] = [1, 0, 0, alpha]  # red
        ax.voxels(data, facecolors=colors)
        return ax
        # plt.show()

    def get_volume(self):
        return self.__x ** 3


class Sphere(ThreeDShape):

    @classmethod
    def get_shape_name(cls):
        return 'Sphere'

    def __init__(self, r):
        super().__init__()
        self.__r = r

    def draw_shape(self, ax):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = self.__r * np.outer(np.cos(u), np.sin(v))
        y = self.__r * np.outer(np.sin(u), np.sin(v))
        z = self.__r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)
        return ax
        # plt.show()

    def get_area(self):
        return 4 * np.pi * self.__r ** 2

    def get_volume(self):
        return 4 / 3 * np.pi * self.__r ** 3


class Parallelepiped(ThreeDShape):

    @classmethod
    def get_shape_name(cls):
        return 'Parallelepiped'

    def __init__(self, a, b, c):
        super().__init__()
        self.__a = a
        self.__b = b
        self.__c = c

    def draw_shape(self, ax):
        Z = np.array([[0, 0, 0],
                      [self.__a, 0, 0],
                      [self.__a, self.__b, 0],
                      [0, self.__b, 0],
                      [0, 0, self.__c],
                      [self.__a, 0, self.__c],
                      [self.__a, self.__b, self.__c],
                      [0, self.__b, self.__c]])

        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax
        # plt.show()

    def get_area(self):
        return 2 * (self.__a * self.__b + self.__b * self.__c + self.__a * self.__c)

    def get_volume(self):
        return self.__a * self.__b * self.__c


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float('inf'), float('inf'))
    return (x / z, y / z)


class Pyramid(ThreeDShape):

    @classmethod
    def get_shape_name(cls):
        return 'Pyramid'

    def __init__(self, a, b, c):
        super().__init__()
        self.__a = a
        self.__b = b
        self.__c = c

    def draw_shape(self, ax):
        # vertices of a pyramid
        points = []
        intersect_point = get_intersect([0, 0], [self.__a, self.__b], [self.__a, 0], [0, self.__b])

        v = np.array([[0, 0, 0], [self.__a, 0, 0], [self.__a, self.__b, 0], [0, self.__b, 0],
                      [intersect_point[0], intersect_point[1], self.__c]])
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

        # generate list of sides' polygons of our pyramid
        verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
                 [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        return ax
        # plt.show()

    def get_area(self):
        return self.__a ** 2 + 2 * self.__a * (self.__b ** 2 - (self.__a ** 2 / 4)) ** 0.5

    def get_volume(self):
        return 1 / 3 * (self.__a * self.__b) * self.__c


class Cylinder(ThreeDShape):

    def __init__(self, r, h):
        super().__init__()
        self.__r = r
        self.__h = h

    @classmethod
    def get_shape_name(cls):
        return 'Сylinder'

    def __data_for_cylinder_along_z(self):
        z = np.linspace(0, self.__h, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = self.__r * np.cos(theta_grid)
        y_grid = self.__r * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    def draw_shape(self, ax):
        Xc, Yc, Zc = self.__data_for_cylinder_along_z()
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
        return ax
        # plt.show()

    def get_area(self):
        return 2 * np.pi * self.__r * (self.__h + self.__r)

    def get_volume(self):
        return np.pi * self.__r ** 2 * self.__h


class Cone(ThreeDShape):

    def __init__(self, r, h):
        super().__init__()
        self.__r = r
        self.__h = h

    @classmethod
    def get_shape_name(cls):
        return 'Cone'

    def __find_color_for_point(self, pt):

        c_x, c_y, c_z = pt

        angle = np.arctan2(c_x, c_y) * 180 / np.pi

        if (angle < 0):
            angle = angle + 360

        if c_z < 0:
            l = 0.5 - abs(c_z) / 2
            # l=0
        if c_z == 0:
            l = 0.5
        if c_z > 0:
            l = (1 - (1 - c_z) / 2)

        if c_z > 0.97:
            l = (1 - (1 - c_z) / 2)

        col = colorsys.hls_to_rgb(angle / 360, l, 1)

        return col

    def draw_shape(self, s):
        n = 64
        xs = np.linspace(-self.__r, self.__r, n)
        ys = np.linspace(-self.__r, self.__r, n)
        r, h = self.__r, self.__h
        zs = np.array([max(h * (1 - np.hypot(x, y) / r), 0)
                       for x in xs for y in ys]).reshape((n, n))
        mxs, mys = np.meshgrid(xs, ys)
        s.plot_surface(mxs, mys, zs)
        s.set_xlabel("x")
        s.set_ylabel("y")
        s.set_zlabel("z")
        return s
        # plt.show()

    def get_area(self):
        return math.pi * self.__r * (self.__r + (self.__r ** 2 + self.__h ** 2) ** 0.5)

    def get_volume(self):
        return 1 / 3 * np.pi * self.__r ** 2 * self.__h


class Triangle(Polygon):

    @classmethod
    def get_shape_name(cls):
        return 'Triangle'

    def __init__(self, side_a, side_b):
        super().__init__()
        self.__a = side_a
        self.__b = side_b

    def get_a(self):
        return self.__a

    def set_a(self, side_a):
        self.__a = side_a

    def get_b(self):
        return self.__b

    def set_b(self, b):
        self.__b = b

    def get_area(self):
        return 1 / 2 * (self.__a * self.__b)

    def __get_points(self):
        points = []
        points.append((0, 0))
        points.append((0, self.__a))
        points.append((self.__b, 0))
        points.append((0, 0))
        return points

    def draw_shape(self, axes):
        points = self.__get_points()
        shape_name = matplotlib.patches.Polygon(points, fill=True)
        axes.add_patch(shape_name)
        return axes
        # plt.show()

    def get_hypot(self):
        return np.hypot(self.__a, self.__b)


class Trapezoid(Polygon):

    @classmethod
    def get_shape_name(cls):
        return 'Trapezoid'

    def __init__(self, a, b, h):
        super().__init__()
        self.__a = a
        self.__b = b
        self.__h = h

    # def __height(self):
    #     return (self.__c ** 2 - 1 / 4 * ((self.__c ** 2 - self.d) ** 2)) ** 0.5

    # def get_area(self):
    #     return ((self.__a + self.__b) / 2) * self.__height()

    def __get_points(self):
        points = []
        c = (self.__a - self.__b) / 2
        if self.__a > self.__b:
            x0 = (0, 0)
            points.append(x0)
            points.append((x0[0] + self.__a, 0))
            points.append((self.__a - c, self.__h))
            points.append((c, self.__h))
        else:
            x0 = (-c, 0)
            points.append(x0)
            points.append((x0[0] + self.__a, 0))
            points.append((self.__a - c * 2, self.__h))
            points.append((0, self.__h))
        points.append(x0)
        return points

    def draw_shape(self, axes):
        shape_name = matplotlib.patches.Polygon(self.__get_points(), fill=True)
        axes.add_patch(shape_name)
        return axes
        # plt.show()

    def get_area(self):
        return (self.__a + self.__b) / 2 * self.__h

    def middle_line(self):
        return (self.__a + self.__b) / 2


class Rhombus(Polygon):

    @classmethod
    def get_shape_name(cls):
        return 'Rhombus'

    def __init__(self, d1, d2):
        super().__init__()
        self.__d1 = d1
        self.__d2 = d2

    def get_area(self):
        return (self.__d1 * self.__d2) / 2

    def __get_points(self):
        points = []
        x0 = (center_xy - self.__d1 / 2, center_xy)
        points.append(x0)
        points.append((center_xy, center_xy + self.__d2 / 2))
        points.append((center_xy + self.__d1 / 2, center_xy))
        points.append((center_xy, center_xy - self.__d2 / 2))
        points.append(x0)
        return points

    def draw_shape(self, axes):
        shape_name = matplotlib.patches.Polygon(self.__get_points(), fill=True)
        axes.add_patch(shape_name)
        return axes
        # plt.show()

    def get_side(self):
        return (self.__d1 ** 2 + self.__d2 ** 2) ** 0.5 / 2


def draw_shape(shape_name, parameters, error):
    shape = Shape.get_shape(shape_name, parameters, error)
    if shape:
        axes = shape.get_axes(parameters)
        axes = shape.draw_shape(axes)
        return axes
        # plt.show()


def get_window_input_label(shape_name):
    if shape_name == 'Circle' or shape_name == 'Sphere':
        return 'Input radius'
    if shape_name == 'Square' or shape_name == 'Cube':
        return 'Input length of side a'
    if shape_name == 'Rectangle' or shape_name == 'Triangle':
        return 'Input lengths of sides a, b'
    if shape_name == 'Rhombus':
        return 'Input diagonals d1, d2'
    if shape_name == 'Trapezoid' or shape_name == 'Parallelepiped' or shape_name == 'Pyramid':
        return 'Input length of sides a, b and height h'
    if shape_name == 'Сylinder' or shape_name == 'Cone':
        return 'Input radius r and height h'


def get_window_evaluation_text(shape):
    shape_name = shape.get_shape_name()
    text = f'Area of figure = {round(shape.get_area(), 4):.4f}'
    if shape_name in ['Square', 'Rectangle']:
        text += f'\nPerimeter: {round(shape.get_perimeter(), 4):.4f}'
    elif shape_name in ['Circle']:
        text += f'\nLength of circle: {round(shape.get_circumference(), 4):.4f}'
    elif shape_name == 'Rhombus':
        text += f'\nLength of side: {round(shape.get_side(), 4):.4f}'
    elif shape_name == 'Trapezoid':
        text += f'\nLength of middle line: {round(shape.middle_line(), 4):.4f}'
    elif shape_name == 'Triangle':
        text += f'\nHypotenuse: {round(shape.get_hypot(), 4):.4f}'
    elif isinstance(shape, ThreeDShape):
        text += f'\nVolume: {round(shape.get_volume(), 4)}'

    return text


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure(figure):
    figure.get_tk_widget().forget()
    plt.close('all')


shape_combo = ['Rectangle', 'Circle', 'Square', 'Triangle', 'Rhombus', 'Trapezoid', 'Sphere',
               'Cube', 'Parallelepiped', 'Pyramid', 'Сylinder', 'Cone']
layout = [[sg.Text('Choose figure:'), sg.InputCombo(shape_combo, 'Rectangle', enable_events=True, key='-SHAPE-')],
          [sg.Text('Input length of sides a, b', key='-TEXT-'), sg.InputText(key='-INPUT_DATA-', size=(20, 20))],
          [sg.Text('(all values - positive integers number separated by commas)')],
          [sg.Button('OK')],
          [sg.Canvas(size=(640, 480), background_color='white', key='canvas')],
          [sg.Text(visible=False, key='-EVALUATION_TEXT-')],
          [sg.Text('Incorrect input!', visible=False, key='-INCORRECT_INPUT-', text_color='yellow')]]

window = sg.Window('Geometric Calculator', layout, size=(710, 720), font='20px', margins=(30, 30))

figure_agg = None

while True:
    event, values = window.read()
    window['-INCORRECT_INPUT-'].update(visible=False)
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break
    shape_name = values['-SHAPE-']
    if event == '-SHAPE-':
        window['-INPUT_DATA-'].update(value='')
        input_label = get_window_input_label(shape_name)
        window['-TEXT-'].update(input_label)

    if event == 'OK':
        shape_parameters = Shape.get_shape_parameters(values['-INPUT_DATA-'].split(','), window['-INCORRECT_INPUT-'])
        if shape_parameters:
            shape = Shape.get_shape(shape_name, shape_parameters, window['-INCORRECT_INPUT-'])
            if shape:
                if figure_agg:
                    delete_figure(figure_agg)
                ax = draw_shape(shape_name, shape_parameters, window['-INCORRECT_INPUT-'])
                figure_agg = draw_figure(window['canvas'].TKCanvas, ax.figure)
                evaluation_text = get_window_evaluation_text(shape)
                window['-EVALUATION_TEXT-'].update(evaluation_text,
                                                   visible=True)
            else:
                window['-INCORRECT_INPUT-'].update(visible=True)
        else:
            window['-INCORRECT_INPUT-'].update(visible=True)

window.close()
