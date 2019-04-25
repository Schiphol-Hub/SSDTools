import copy
import io
import os
import re
import textwrap
import numpy as np
import pandas as pd
import shapefile
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Set the gelijkwaardigheidscriteria
gwc = {'doc29_2005': [13600, 166500, 14600, 45000],
       'doc29_2015': [14000, 180000, 14800, 48500],
       'doc29_2018': [12000, 186000, 12800, 50000]}


def extract_year_from_file_name(file_name):
    """
    The default method to obtain the year from a provided file name.

    :param str file_name: The name of the file.
    :rtype int|None
    """

    # Extract the matches
    matches = re.findall(r'y\d{4}', file_name)

    # Check if there are any matches
    if len(matches) < 1:
        return None

    # Take the first match and extract the year
    return int(matches[0][1:])


def meteotoeslag_years(method, unit):
    """
    Determine the years to include in the meteotoeslag based on the provided method.

    :param str method: The method for selecting the meteorological representative years, which is either 'empirisch'
     or 'hybride'.
    :param str unit: The noise level unit, which is either 'Lden' or 'Lnight'.
    :return: the years to include in the meteotoeslag.
    :rtype np.ndarray
    """

    # Set the exceptional years for each method and unit combination
    exceptional_years = {
        ('empirisch', 'Lden'): [1972, 1976, 1981, 1990, 1994, 1996, 2000, 2003],
        ('empirisch', 'Lnight'): [1973, 1979, 1985, 1989, 1994, 1995, 1996, 2002],
        ('hybride', 'Lden'): [1981, 1984, 1993, 1994, 1996, 2000, 2002, 2010],
        ('hybride', 'Lnight'): [1973, 1976, 1980, 1987, 1994, 1995, 1996, 2010]
    }

    # Exclude the years from the default range of 1971-2011 based on the method and unit
    return np.setdiff1d(np.arange(1971, 2011), exceptional_years[(method, unit)])


class Grid(object):
    """
    A Grid object contains the data and methods related to noise grids.

    There are two variants:
    1) A single contour grid
    2) A multi-contour grid

    Multi-contour grids are use to indicate the certainty ranges of the noise levels for a given traffic scenario.
    """

    def __init__(self, data=None, info=None, shape=None, years=None, unit=None):
        """

        :param list(np.ndarray)|np.ndarray data: grid data, is two-dimensional for single contour grids and
        three-dimensional for multi-contour grids.
        :param list(dict)|dict info: grid information

        todo: Create a format specification for the grid information.

        """

        if data is not None:
            self.data = data
        if info is not None:
            self.info = info
            self.shape = Shape(info if isinstance(info, dict) else info[0])
        if shape is not None:
            self.shape = shape
        if years is not None:
            self.years = years
        if unit is not None:
            self.unit = unit

        self.validate(exclude=['datum', 'tijd', 'nvlb'])

    @classmethod
    def read_envira(cls, path):
        """
        Create a Grid object from an envira file.

        :param str path: The path to the envira file.
        :rtype Grid
        """

        # Read the envira file
        info, data = read_envira(path)

        # Extract the unit from the envira file
        unit = info['eenheid']

        # Return the object
        return cls(data=data, info=info, unit=unit)

    @classmethod
    def read_enviras(cls, path, pattern='*.dat', year_extractor=extract_year_from_file_name):
        """
        Create a Grid object from multiple envira files.

        :param str path: The path to the envira files.
        :param str pattern: The pattern used to match the envira files.
        :param function year_extractor: The method used to extract the year from the file name.
        :rtype Grid
        """

        # Get the envira files
        file_paths = [os.path.join(path, f) for f in os.listdir(path) if re.search(pattern, f)]

        # Create info and data lists
        cls_info = []
        cls_data = []
        cls_years = []

        # Read the envira files
        for file_path in file_paths:
            # Extract the data and header from the file
            info, data = read_envira(file_path)

            # Extract the year from the file path
            year = year_extractor(file_path)

            # Put the extracted data in the lists
            cls_info.append(info)
            cls_data.append(data)
            cls_years.append(year)

        # Extract the unit from the first envira file
        unit = cls_info[0]['eenheid']

        # Add the data to a Grid object
        return cls(data=cls_data, info=cls_info, unit=unit, years=cls_years)

    def validate(self, exclude=None):
        """
        Validate if the object's data and info are consistent with each other.
        Only checks if multigrids are consistent among each other.

        """

        # Create an empty list if exclude is not provided
        exclude = [] if exclude is None else exclude

        if isinstance(self.data, list) and isinstance(self.info, list):
            # This is the case for multigrid

            # Check if the lists are equal
            if len(self.data) != len(self.info):
                raise IndexError('Provided data list and info list should have the same length.')

            # Set the elements to check
            data = np.array(self.data[0])

            # Put all the info in a data frame for easy checking
            info = pd.DataFrame(self.info)

            if not info.duplicated(subset=info.columns[~info.columns.isin(exclude)], keep=False).all():
                raise ValueError('All info in the provided info list should be the same')

        elif isinstance(self.data, list) or (hasattr(self, 'info') and isinstance(self.info, list)):
            raise TypeError('Supplied data and info for a multigrid should both be lists.')

        elif isinstance(self.data, np.ndarray) and hasattr(self, 'years') and isinstance(self.years, list):
            # This is the case for a meteotoeslag grid
            pass
        else:
            # This is the case for a normal grid
            if not (self.shape.y_number, self.shape.x_number) == self.data.shape:
                raise IndexError('Provided data does not have the same shape as mentioned in the header file.')

    def copy(self):
        return copy.deepcopy(self)

    def to_envira(self, path):
        """

        :param str path:
        """

        # Update the info with the current shape
        self.info.update(self.shape.to_dict())

        # Write the data to the selected path
        return write_envira(path, self.info, self.data)

    def to_shapefile(self, path, level):

        # Extract the x and y coordinates
        x = self.shape.get_x_coordinates()
        y = self.shape.get_y_coordinates()

        # Create a contour using matplotlib without opening a figure
        figsize = (21 / 2.54, 21 / 2.54)
        fig = plt.figure(figsize=figsize)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        plt.close(fig)
        ax.contourf(x, y, self.data, levels=[level, 100], colors='blue', linewidths=.2)
        cs = ax.contour(x, y, self.data, levels=[level], colors='green', linewidths=2)

        # Extract coordinates from contour and put into correct format for shapefile
        contours = cs.allsegs[0]

        # Create polygons with correct data type (and reverse the coordinates to get rid of open holes)
        flat_list = [contour_coordinates[::-1, :].tolist() for contour_coordinates in contours]

        # Create the shapefile
        w = shapefile.Writer(target=path, shapeType=shapefile.POLYGON)
        w.poly(flat_list)
        w.field('FIRST_FLD', 'C', '40')
        w.field('SECOND_FLD', 'C', '40')
        w.record('First', 'Polygon')

    def scale(self, factor):
        """
        Apply a scaling factor to the data.

        :param float|int factor: the factor to be applied.
        """

        if isinstance(self.data, list):
            self.data = [d + 10 * np.log10(factor) for d in self.data]
        else:
            self.data += 10 * np.log10(factor)

        return self

    def gelijkwaardigheid(self):
        """
        todo: Translate gelijkwaardigheid
        todo: Add the functionality of GPtools_matlab/lib/GPcalc_gelijkwaardigheid.m here
        """
        pass

    def inpasbaarvolume(self):
        """
        todo: Translate inpasbaarvolume
        todo: Add GPtools_matlab/lib/GPcalc_inpasbaarvolume.m here
        """
        pass

    def contourpunten(self):
        """
        todo: Translate contourpunten
        todo: Add GPtools_matlab/lib/ContourPunten.m here
        """
        pass

    def hg(self):
        """
        Calculate the Hoeveelheid Geluid (HG).

        todo: Add support for multigrid.
        """

        if isinstance(self.data, list):
            raise TypeError('Hoeveelheid Geluid (HG) cannot be calculated for a multi-contour grid.')

        # Conversion to "Hindersom" without scaling
        hs = 10 ** (self.data / 10.)

        # Return total noise level (HG)
        return 10. * np.log10(hs.sum() / np.array(hs.shape).prod())

    def meteotoeslag_from_method(self, method):
        """
        Calculate the meteotoeslag based on the provided method.

        :param str method: The method for selecting the meteorological representative years, which is either 'empirisch'
         or 'hybride'.
        :return the max-grid and the included meteorological years.
        :rtype tuple(np.ndarray, np.ndarray)
        """

        return self.meteotoeslag_from_years(meteotoeslag_years(method, self.unit))

    def meteotoeslag_grid_from_method(self, method):
        """
        Calculate the meteotoeslag based on the provided method.

        :param str method: The method for selecting the meteorological representative years, which is either 'empirisch'
         or 'hybride'.
        :return the max-grid and the included meteorological years.
        :rtype tuple(np.ndarray, np.ndarray)
        """

        # Calculate the grid with meteorological surcharge and the included years
        meteotoeslag, meteo_years = self.meteotoeslag_from_years(meteotoeslag_years(method, self.unit))

        return Grid(data=meteotoeslag, unit=self.unit, years=meteo_years, shape=self.shape)

    def meteotoeslag_from_years(self, years):
        """
        Determine the meteotoeslag excluding the extraordinary meteorological years.
        Meteotoeslag is a surcharge for meteorological conditions and is a max-grid based on the provided grids

        :param list|np.ndarray years: the years to include in the meteotoeslag, should include
        :return the max-grid and the included meteorological years.
        :rtype tuple(np.ndarray, np.ndarray)
        """

        # Get the selected years from the provided years of the grid
        selected_years = np.isin(self.years, years)

        # There should be 32 years to include
        if selected_years.sum() != 32:
            raise LookupError(
                'Expected 32 years for the meteorological surcharge but found {} years'.format(selected_years.sum()))

        # Get the noise levels for all years
        data = np.array(self.data)

        # Select the maximum noise levels based on the selected years
        meteorological_surcharge = np.amax(data[selected_years, :, :], axis=0)

        # Return the grid with meteorological surcharge and the included years
        return meteorological_surcharge, years

    def statistics(self):
        """
        Determine the average, standard deviation and the confidence interval for a multigrid.

        :return collection of statistics for the current data.
        :rtype dict(nd.array)
        """

        if not isinstance(self.data, list):
            raise TypeError('Statistics can only be extracted from multigrids')

        # Convert the data to a multidimensional array
        data = np.array(self.data)

        # Extract the statistics for each point in the grid
        mean = 10 * np.log10(np.mean(10 ** (data / 10.), axis=0))
        standard_deviation = np.std(self.data, axis=0)

        # Calculate the 99.5% confidence interval
        z = 2.5758
        upper_bound_confidence_interval = mean + z * standard_deviation
        lower_bound_confidence_interval = mean - z * standard_deviation

        return {
            'mean': Grid(data=mean, shape=self.shape, unit=self.unit),
            'std': Grid(data=standard_deviation, shape=self.shape, unit=self.unit),
            'dhi': Grid(data=upper_bound_confidence_interval, shape=self.shape, unit=self.unit),
            'dlo': Grid(data=lower_bound_confidence_interval, shape=self.shape, unit=self.unit),
        }

    def grid_from_year(self, year):
        """
        Determine grid for the required year
        :param year: int
        :return: Grid
        :rtype: Grid object
        """
        if not isinstance(self.data, list):
            raise TypeError('Grid from year can only be extracted from multigrids')

        # Get the location of the requested year, select the first match
        index = np.where(np.array(self.years) == year)[0][0]

        # Return the object
        return Grid(data=self.data[index], info=self.info[index], unit=self.unit)

    def interpolation_function(self):
        """
        Determine the bi-cubic spline interpolation function.
        :return the interpolation function.
        :rtype function
        """

        if isinstance(self.data, list):
            raise TypeError('Interpolation functions can only be created from single grids.')

        # Extract the coordinates of the current grid
        x = self.shape.get_x_coordinates()
        y = self.shape.get_y_coordinates()

        # Extract the data of the current grid
        z = self.data

        # Return the bi-cubic spline interpolation function
        return RectBivariateSpline(y, x, z)

    def refine(self, factor):
        """
        Refine the grid with a bi-cubic spline interpolation.
        :return refined grid
        :rtype Grid object
        """

        # Refine the current shape
        shape = self.shape.copy().refine(factor)

        # Return a reference to this object
        return self.resize(shape)

    def resize(self, shape):
        """
        Reshape the grid based on with a bi-cubic spline interpolation.
        :return resized grid
        :rtype Grid object
        """

        if isinstance(self.data, list):
            # Refine the grids of this multigrid and update the data and info
            grids = [self.grid_from_year(year).resize(shape) for year in self.years]
            self.data = [grid.data for grid in grids]
            self.info = [grid.info for grid in grids]
        else:
            # Refine the grid and update the data and info
            self.data = self.interpolation_function()(shape.get_y_coordinates(), shape.get_x_coordinates())
            if hasattr(self, 'info'):
                self.info.update(shape.to_dict())

        # Assign the shape to the object
        self.shape = shape

        return self

    def scale_per_time_interval(self, night_grid, scale_de=1, scale_n=1, apply_lnight_time_correction=True):
        """
        Scale the Lden-grid separately for the day- and evening period and the night period.

        todo: Add support for multigrid.

        :param Grid night_grid: the Lnight grid.
        :param float scale_de: the scaling factor for day- and evening.
        :param float scale_n: the scaling factor for night.
        :param bool apply_lnight_time_correction: setting for the Lnight time correction, defaults to True.
        :rtype: Grid
        """

        if self.unit != "Lden":
            raise TypeError('The supplied base grid to scale should have the unit Lden.')
        if night_grid.unit != "Lnight":
            raise TypeError('The supplied night grid should have the unit Lnight.')
        if isinstance(self.data, list) or isinstance(night_grid.data, list):
            raise TypeError('This method does not support multigrids.')

        # Convert Lnight to Ln
        n_grid = night_grid.copy().scale(10)

        # Apply a time correction for Lnight if requested
        if apply_lnight_time_correction:
            n_grid.scale(8 / 24.)

        # Scale only the day- and evening
        self.data = 10 * np.log10(
            (10 ** (self.data / 10.) - 10 ** (n_grid.data / 10.)) * scale_de
            + (10 ** (n_grid.data / 10.) * scale_n))

        # Return itself, the scaled grid
        return self


def relative_den_norm_performance(scale, norm, wbs, den_grid, night_grid=None, scale_de=None, scale_n=None,
                                  apply_lnight_time_correction=True):
    """
    Calculate the difference with respect to the provided norm for affected houses and annoyed people.

    This function can be used to efficiently determine the traffic volume that fits within the gwc bounds. To do this
    one can use the scipy.brentq routine. An example is shown below:

        factor = brentq(relatief_norm_etmaal, 1.0, 3.0, rtol=0.0001, args=(norm, wbs, den_grid))

    :param float scale: the scale to apply.
    :param dict norm: the norm to match
    :param WBS wbs: the woningbestand.
    :param Grid den_grid: the Lden grid.
    :param Grid night_grid: the Lnight grid, as arguments for den_grid.scale_per_time_interval()
    :param float scale_de: see Grid.scale_per_time_interval()
    :param float scale_n: see Grid.scale_per_time_interval()
    :param bool apply_lnight_time_correction: see Grid.scale_per_time_interval()
    :return the available room in terms of houses or annoyed people.
    """

    if night_grid is not None:
        if scale_de is None:
            scale_de = scale
        if scale_n is None:
            scale_n = scale

        # Apply the scale per time interval
        grid = den_grid.copy().scale_per_time_interval(night_grid=night_grid, scale_de=scale_de, scale_n=scale_n,
                                                       apply_lnight_time_correction=apply_lnight_time_correction)
    else:
        # Apply the scale
        grid = den_grid.copy().scale(scale)

    # Add the Lden data to the wbs
    wbs = wbs.copy().add_noise_from_grid(grid)

    # Count the number of homes
    w = wbs.count_homes_above(58, 'Lden')

    # Count the number of annoyed people
    p = wbs.count_annoyed_people(48)

    # Get the difference with the norm
    delta_w = norm[0] - w
    delta_p = norm[1] - p

    # Return the lowest value
    return min(delta_w, delta_p)


class Shape(object):
    def __init__(self, data):
        self.x_start = data['x_start']
        self.x_stop = data['x_stop']
        self.x_step = data['x_step']
        self.x_number = data['x_number']
        self.y_start = data['y_start']
        self.y_stop = data['y_stop']
        self.y_step = data['y_step']
        self.y_number = data['y_number']

    def refine_x(self, factor):
        """
        Refine a shape by increasing the x resolution by factor
        :param factor: should be a positive real number
        :return: refined shape
        :rtype: Shape object
        """
        if 0 < factor < np.inf:

            # Calculate the new step distance
            self.x_step /= factor

            # Calculate the new number of steps
            self.x_number = 1 + factor * (self.x_number - 1)
        else:
            raise ValueError('Invalid input: Factor must be a positive number')

        return self

    def refine_y(self, factor):
        """
        Refine a grid by increasing the y resolution by factor
        :param factor: should be a positive real number
        :return: refined grid
        :rtype: Shape object
        """
        if 0 < factor < np.inf:

            # Calculate the new step distance
            self.y_step /= factor

            # Calculate the new number of steps
            self.y_number = 1 + factor * (self.y_number - 1)
        else:
            raise ValueError('Invalid input: Factor must be a positive number')

        return self

    def refine(self, factor):
        """
        Refine a grid by increasing the x and y resolution by factor
        :param factor: should be a positive real number
        :return: refined shape
        :rtype: Shape object
        """
        if 0 < factor < np.inf:
            self.refine_x(factor)
            self.refine_y(factor)
        else:
            raise ValueError('Invalid input: Factor must be a positive number')

        return self

    def set_x_number(self, number):
        """
        Set the number of points in x direction for the shape, adjust shape accordingly
        :param number: should be a real number greater than 2
        :return: adjusted shape
        :rtype: Shape object
        """
        if 2 <= number < np.inf:
            self.x_number = number
            self.x_step = (self.x_stop - self.x_start) / (self.x_number - 1)
        else:
            raise ValueError('Cannot set the number of x coordinates to {}. The minimum number is 2.'.format(number))

        return self

    def set_y_number(self, number):
        """
        Set the number of points in y direction for the shape, adjust shape accordingly
        :param number: should be a real number greater than 2
        :return: adjusted shape
        :rtype: Shape object
        """
        if 2 <= number < np.inf:
            self.y_number = number
            self.y_step = (self.y_stop - self.y_start) / (self.y_number - 1)
        else:
            raise ValueError('Cannot set the number of x coordinates to {}. The minimum number is 2.'.format(number))

        return self

    def set_x_step(self, step):
        """
        Set step size in x direction for the shape, adjust shape accordingly
        :param step: should be a positive real number
        :return: adjusted shape
        :rtype: Shape object
        """
        if 0 < step < np.inf:
            self.x_step = step
            self.x_number = 1 + (self.x_stop - self.x_start) / self.x_step
        else:
            raise ValueError('Invalid input: step must be a positive number')

        return self

    def set_y_step(self, step):
        """
        Set step size in y direction for the shape, adjust shape accordingly
        :param step: should be a positive real number
        :return: adjusted shape
        :rtype: Shape object
        """
        if 0 < step < np.inf:
            self.y_step = step
            self.y_number = 1 + (self.y_stop - self.y_start) / self.y_step
        else:
            raise ValueError('Invalid input: step must be a positive number')

        return self

    def get_x_coordinates(self):
        """
        Extract x coordinates from shape
        :return: array of x coordinates
        :rtype: numpy array
        """
        return np.linspace(self.x_start, self.x_stop, num=self.x_number)

    def get_y_coordinates(self):
        """
        Extract y coordinates from shape
        :return: array of y coordinates
        :rtype: numpy array
        """
        return np.linspace(self.y_start, self.y_stop, num=self.y_number)

    def to_dict(self):
        return self.__dict__

    def copy(self):
        return copy.deepcopy(self)


def hdr_val(string, type):
    """
    Read value from header line
    """
    name, val = string.split()
    return type(val)


def read_envira(file_path):
    """
    Read NLR grid-file and return header and noise data

    todo: create a dict or specification for the header
    """

    with open(file_path, "r") as file:
        # Create an empty dict for the header
        header = dict()

        header['tekst1'] = file.readline().strip()
        header['tekst2'] = file.readline().strip()
        header['tekst3'] = file.readline().strip()

        header['datum'], header['tijd'] = file.readline().split()

        header['eenheid'] = hdr_val(file.readline(), str)
        header['grondinvloed'] = hdr_val(file.readline(), str)

        # skip, following line, because it is not used anymore
        # hdr['tellingen'] = hdr_val(data.readline(), int)
        file.readline()

        header['demping_landing'] = hdr_val(file.readline(), float)
        header['demping_start'] = hdr_val(file.readline(), float)
        header['mindba'] = hdr_val(file.readline(), float)
        header['tijdstap'] = hdr_val(file.readline(), float)
        header['x_start'] = hdr_val(file.readline(), int)
        header['x_stop'] = hdr_val(file.readline(), int)
        header['x_step'] = hdr_val(file.readline(), int)
        header['x_number'] = hdr_val(file.readline(), int)
        header['y_start'] = hdr_val(file.readline(), int)
        header['y_stop'] = hdr_val(file.readline(), int)
        header['y_step'] = hdr_val(file.readline(), int)
        header['y_number'] = hdr_val(file.readline(), int)
        header['nvlb'] = hdr_val(file.readline(), int)
        header['neff'] = hdr_val(file.readline(), float)
        header['nlos'] = hdr_val(file.readline(), int)
        header['nweg'] = hdr_val(file.readline(), int)

        # Overwrite unreliable values
        header['x_stop'] = header['x_start'] + (header['x_number'] - 1) * header['x_step']
        header['y_stop'] = header['y_start'] + (header['y_number'] - 1) * header['y_step']

        # Extract the noise data from the remaining lines
        data = np.fromfile(file, sep=" ")

        # Check if the provided header and data are compatible
        if data.shape[0] != header['y_number'] * header['x_number']:
            raise ValueError('The header of the envira file is not consistent with its data.')

        # Reshape the data
        data = np.flipud(np.resize(data, (header['y_number'], header['x_number'])))

    return header, data


def write_envira(file_path, hdr, dat):
    """
    Write NLR grid-file with header and noise data
    """

    with open(file_path, 'w') as f:
        # Write the header
        f.write('{:s}\n'.format(hdr['tekst1']))
        f.write('{:s}\n'.format(hdr['tekst2']))
        f.write('{:s}\n'.format(hdr['tekst3']))
        f.write('{:s} {:s}\n'.format(hdr['datum'], hdr['tijd']))
        f.write('EENHEID {:s}\n'.format(hdr['eenheid']))
        f.write('GRONDINVLOED {:s}\n'.format(hdr['grondinvloed']))
        f.write('TELLINGEN\n')
        f.write('DEMPING-LANDING {:6.2f}\n'.format(hdr['demping_landing']))
        f.write('DEMPING-START {:6.2f}\n'.format(hdr['demping_start']))
        f.write('MINDBA {:6.2f}\n'.format(hdr['mindba']))
        f.write('TIJDSTAP {:6.2f}\n'.format(hdr['tijdstap']))
        f.write('X-ONDER {:9.0f}\n'.format(hdr['x_start']))
        f.write('X-BOVEN {:9.0f}\n'.format(hdr['x_stop']))
        f.write('X-STAP {:9.0f}\n'.format(hdr['x_step']))
        f.write('NX{:6.0f}\n'.format(hdr['x_number']))
        f.write('Y-ONDER {:9.0f}\n'.format(hdr['y_start']))
        f.write('Y-BOVEN {:9.0f}\n'.format(hdr['y_stop']))
        f.write('Y-STAP {:9.0f}\n'.format(hdr['y_step']))
        f.write('NY{:6.0f}\n'.format(hdr['y_number']))
        f.write('NVLB {:9.0f}\n'.format(hdr['nvlb']))
        f.write('NEFF {:9.0f}\n'.format(hdr['neff']))
        f.write('NLOS {:9.0f}\n'.format(hdr['nlos']))
        f.write('NWEG {:9.0f}'.format(hdr['nweg']))

        # Create an in-memory stream for text I/O to capture the numpy data as a string
        z = io.StringIO()
        np.savetxt(z, np.flipud(dat), fmt='%12.6E')
        x = z.getvalue()
        z.close()

        # Extract the individual lines except the last line which is empty
        s = x.split('\n')[:-1]

        # Create a line wrapper
        wrapper = textwrap.TextWrapper(initial_indent=' ', subsequent_indent=' ', width=131)

        # Write each line to the data file
        [f.write('\n' + '\n'.join(wrapper.wrap(p))) for p in s]
