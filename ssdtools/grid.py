import copy
import io
import os
import re
import textwrap
import numpy as np
import pandas as pd

# import shapefile
###TODO Vincent, zonder deze module werkt de toolbox niet meer, 
###              misschien het script/de module opnemen in de ssdtools?

import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

# Set the gelijkwaardigheidscriteria
gwc = {'doc29_2005': [13600, 166500, 14600, 45000],
       'doc29_2015': [14000, 180000, 14800, 48500],
       'doc29_2018': [12000, 186000, 12800, 50000]}

def read_grid(grid,
              mean='mean',
              noise='Lden',
              pattern=None):
    """
    Import and read dat-, txt- or folders with envira-bestanden into a grid
    
    :param str|Grid grid: string with folder location with multiple envira-files, or text/dat-file
    :param str mean: Average grid type to use for the average noise levels
    :param str noise: The noise level unit, which is either 'Lden' or 'Lnight'.
    :param str pattern: The pattern used to match the envira files.    
    :return: Grid object.
    """
            
    # read noise grid
    if isinstance(grid, str):
        if grid.endswith(('.dat', '.txt')):
            grid = Grid.read_envira(grid)
        else: # folder location is given
            grid = Grid.read_enviras(grid, noise=noise, pattern=pattern)
            if mean == 'mm':
                grid = grid.meteotoeslag_grid_from_method('hybride')
            else:
                grid = grid.statistics()[mean]       
    return grid    

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

    def __init__(self, data=None, info=None, shape=None, years=None, unit=None, unequal_grids=None):
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

        # This is to prevent the validation to throw an error if unequal grids are read on purpose
        if unequal_grids is None:
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
    def read_enviras(cls, 
                     path,
                     noise='Lden',
                     pattern=None,
                     year_extractor=extract_year_from_file_name,
                     unequal_grids=None):
        """
        Create a Grid object from multiple envira files.

        :param str path: The path to the envira files.
        
        :param str pattern: The pattern used to match the envira files.
        :param function year_extractor: The method used to extract the year from the file name.
        :rtype Grid
        """

        # Pattern
        if pattern is None:
            pattern = r'\w[\s-]*' + noise + r'\s?y\d{4}\.dat$'
            
        # Get the envira files
        files = [os.path.join(path, f) for f in os.listdir(path) if re.search(pattern, f)]

        # Create info and data lists
        cls_info = []
        cls_data = []
        cls_years = []

        # Read the envira files
        for file in files:
            # Extract the data and header from the file
            info, data = read_envira(file)

            # Extract the year from the filename
            year = year_extractor(file)

            # Put the extracted data in the lists
            cls_info.append(info)
            cls_data.append(data)
            cls_years.append(year)

        # Extract the unit from the first envira file
        unit = cls_info[0]['eenheid']

        # Add the data to a Grid object
        return cls(data=cls_data, info=cls_info, unit=unit, years=cls_years, unequal_grids=unequal_grids)

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

        # Extract coordinates from contour and put into correct format for shapefile
        contours = self.contour_points(level)

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

    def contour_points(self, level):
        # Extract the x and y coordinates
        x = self.shape.get_x_coordinates()
        y = self.shape.get_y_coordinates()

        # Create a contour using matplotlib without opening a figure
        figsize = (21 / 2.54, 21 / 2.54)
        fig = plt.figure(figsize=figsize)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        plt.close(fig)
        ax.contourf(x, y, self.data, levels=[level, 100], colors='blue')
        cs = ax.contour(x, y, self.data, levels=[level], colors='green', linewidths=2)

        # Extract coordinates from the contour
        return cs.allsegs[0]

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

        return Grid(data=meteotoeslag, unit=self.unit, years=meteo_years, shape=self.shape, info=self.info[0])

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

        :return the interpolation function, with first argument the y-coordinate and second argument the x-coordinate.
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
        ###TODO is die copy nodig?
        shape = self.shape.copy().refine(factor)

        # Return a reference to this object
        return self.copy().resize(shape)

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
        if scale_de <= 0 or scale_n <= 0:
            raise ValueError('This method does not support negative scaling factors.')

        # Convert Lnight to Ln
        n_grid = night_grid.copy().scale(10)

        # Apply a time correction for Lnight if requested
        if apply_lnight_time_correction:
            n_grid.scale(8 / 24.)

        # Scale only the day- and evening
        self.data = 10 * np.log10((10 ** (self.data / 10.) - 10 ** (n_grid.data / 10.)) * scale_de
                                  + (10 ** (n_grid.data / 10.) * scale_n))

        # Return itself, the scaled grid
        return self

    def get_area_from_contour(self, level):
        """
        Calculate the area in a contout with specified level.

        :level: The level of the contour in dB'.
        :rtype: area in km2
        
        TO DO - first check on how to deald with islands an lakes seems to work OK. more verification is needed.
        """
       
        # Extract the x and y coordinates
        x = self.shape.get_x_coordinates()
        y = self.shape.get_y_coordinates()

        # Create a contour using matplotlib without opening a figure
        figsize = (21 / 2.54, 21 / 2.54)
        fig = plt.figure(figsize=figsize)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        plt.close(fig)
        ax.contourf(x, y, self.data, levels=[level, 100], colors='blue')
        cs = ax.contour(x, y, self.data, levels=[level], colors='green', linewidths=2)

        ##organizing paths and computing individual areas
        paths = cs.collections[0].get_paths()
       
        # loop over paths and find area
        area_of_individual_polygons = []
        for p in paths:
              sign = -1 ##<-- assures that area of first(outer) polygon will be summed
              if p.should_simplify:
                     sign = 1  ##<-- assures that inner polygon will be summed
        
              verts = p.vertices
              area_of_individual_polygons.append(sign*Polygon(verts).area)

        ##computing total area and convert to km2      
        total_area = np.sum(area_of_individual_polygons)/1000000
       
        return total_area
   
    def add(self,grid2=None):
        """
        add grids by tranforming to hindersommmen.
        N.B. now support multigrids, by adding them into a single grid

        :grid2 = grid to add to self, only for single grids

        """
        if isinstance(self.data, list):
            # init grid_out:
            grid_out = Grid(data=self.data[0] ,info=self.info[0],unit=self.unit)

            # now loop over grids and add
            for d in self.data:

                data_1=10**(grid_out.data/10)
                data_2=10**(d/10)
                
                data_out=data_1 + data_2
                
                data=10*np.log10(data_out)
                grid_out.data = data
            return grid_out
            
        else:
            if self.unit!=grid2.unit:
                raise ValueError("Grids do not have the same unit!")
             
            if self.shape!=grid2.shape:
                #raise ValueError("Grids do not have the same size")
                print("Grids do not have the same size. The new grid has been resized.")
                
                grid2=grid2.resize(self.shape)
            
            data_1=10**(self.data/10)
            data_2=10**(grid2.data/10)
            
            data_out=data_1 + data_2
            
            data=10*np.log10(data_out)
            self.data = data
        
            return self
    
    def subtract(self,grid2):  
        
        if self.unit!=grid2.unit:
            raise ValueError("Grids do not have the same unit!")
         
        if self.shape!=grid2.shape:
            #raise ValueError("Grids do not have the same size")
            print("Grids do not have the same size. The new grid has been resized.")
            
            grid2=grid2.resize(self.shape)
            
        data_1=10**(self.data/10)
        data_2=10**(grid2.data/10)
        
        data_out=data_1 - data_2
        
        data_out=np.where(data_out<=0, 1, data_out) 
        
        data=10*np.log10(data_out)
        
        self.data = data
        return self
    
    @classmethod
    def read_enviras_NAxx(cls, grid_path, pattern='*.dat', conditions_path=None):
        """
        Create a Grid object from multiple LAmax envira grids.

        :param str grid_path: The path to the envira files.
        :param str pattern: The pattern used to match the envira files.
        :param str conditions_path: The path used to select only the relevant noise contour conditions.
        :rtype Grid
        """

        # Get the envira files
        file_paths = [os.path.join(grid_path, f) for f in os.listdir(grid_path) if re.search(pattern, f)]

        # Create info and data lists
        cls_info = []
        cls_data = []
        cls_years = []
        
        # Check if a conditions file is given to only load the relevant contours
        if conditions_path:
            conditions=pd.read_csv(conditions_path)
           
        # Read the envira files
        for file_path in file_paths:
            
            # These are the names of the noise contours for the scaling factors. 
            #Only called years to not alter the rest of the object too much
            year = file_path.strip(grid_path).strip('.dat').strip('\\')
            
            if conditions_path:
                # If the name of thefile is in the conditions file, then process that file. Else do not use it
                if conditions.isin([year]).any(axis=None)==True:
            
                    # Extract the data and header from the file
                    info, data = read_envira(file_path)
    
                    # Put the extracted data in the lists
                    cls_info.append(info)
                    cls_data.append(data)
                    cls_years.append(year)
                
            # If no conditions file is given, just load all files in the path    
            else:
                 # Extract the data and header from the file
                info, data = read_envira(file_path)

                # Put the extracted data in the lists
                cls_info.append(info)
                cls_data.append(data)
                cls_years.append(year)         
       
        # Unit assigned to satisfy object structure; not actually used for anything
        unit = 'NAxx'

        # Add the data to a Grid object
        # unequal_grids=True as each envira file has a different shape. Issue with DAISY
        return cls(data=cls_data, info=cls_info, unit=unit, years=cls_years, unequal_grids=True)

  
    def create_NAxx(self, number_above_dB, path, reshape_data=None, refinement_factor=1):
        """
        Creates the actual NAxx grid. First it resizes every individual grid to the same 
        
        # NOTE: using the standard self.refine will lead to artifacts in the plots
        # That funcion uses a bivariate cubic spline to interpolate the data. 
        # This does not work well with discrete steps, thus interpolation issus will arise
        # Never use the refine attribute for NAxx data; not in here or in any plot!
        # Use the refinement_factor here, or give a better reshape_data grid for 
        # better results. 
        # For any plotting, always set the refinement_factor in the plotting(!) 
        # calls to 1 to prevent erroneous plots.

        :number_above_dB: The lower cut-off limit in dB for the NAxx grid.
        :path: The file path for the conditions file.
        :reshape_data: Dictionary that specifies the extent and resolution of the resized grids
        :refinement_factor: If no specific reshaoe data is given, then the default size of the basic grid is 
                            used to reshape the data. For a higher resolution, the amount of steps can be refined 
                            with this parameter. From experience, this factor should not be set higher than 10-12
                            to prevent memory problems.
        :rtype: single Grid object

        """
 
        # Conditions file containing flight profile names and scaling factors 
        # is read and converted to numpy array
        self.flights_scale=pd.read_csv(path).to_numpy(dtype=str)
        
        # Resizing the grids to make them all have an equal size
        # If no desired shape dict is given, then use these default values for the reshaping
        # Refinement factor can be used to improve quality of resulting grids
        data={'x_start' : 84000,
              'x_stop' : 155000,
              'x_step' : 500/refinement_factor,
              'x_number' : 143*refinement_factor,
              'y_start' : 455000,
              'y_stop' : 526000,
              'y_step' : 500/refinement_factor,
              'y_number' : 143*refinement_factor}
        
        if reshape_data is not None:
            data=reshape_data

        # Doing the actual reshaping using the Shape class and the resize attribute of Grid
        gridshape=Shape(data)
        self.resize(gridshape)

        # Creating an empty array of zeros in preparation of the scaling factors assignment
        # The multigrid will be multiplied by this, so every layer gets the respective scaling
        # factor from the flights_scale. 
        scale=np.zeros(np.shape(self.data))
        self.scale=scale
        
        # Looping through all individual grids to filter for NAxx and assign scaling
        for i in range(len(self.scale)):
            
            # Filter of the datapoints are above the threshold.
            # If yes, assign a 1, if not, a 0.
            self.data[i][self.data[i]<number_above_dB]=0        
            self.data[i][self.data[i]>=number_above_dB]=1
            
            # Check if the current grid is in the list of conditions
            # If yes, assign scaling factors, if not, assign a 0 scaling to
            # remove the grid completely from the count
            if self.years[i] in self.flights_scale:
                # Find the correct scaling factor to assign to the respective grid
                position=np.where(self.flights_scale==self.years[i])[0][0]
                scale[i,:,:]=float(self.flights_scale[position,1])
                
            # if grid name is not in conditions file, remove grid from count    
            else:
                scale[i,:,:]=0
                
        # Apply the scaling factor matrix to the 0/1 multigrid
        new_grids=self.data*scale
        
        # Sum all values along the 3rd axis to create sum grid
        self.data=new_grids.sum(axis=0)
        
        # Two verification checkt to compare which grids are present, 
        # and which values are assigned to them. TO enable checking that the correct
        # ones were set to 0
        self.years_check=self.years
        self.scale_check=self.scale[:,0,0]
        
        
        #These are just here to make the object work. Not actually used for anything
        self.info=self.info[0]
        self.years=self.years[-1]
        
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

    @classmethod
    def read_echo_api(cls, grid_id: int, api_client=None):
        """
        Method to retrieve a Shape object from the Echo API.

        :param api_client:
        :return:
        """
        import echo_api_sdk

        # Create an instance of the GridsAPI class
        api = echo_api_sdk.GridsApi(api_client)

        # Get the echo_api.sdk.Grid
        shape = api.get_grid(grid_id)

        # Convert to ssdtools.Shape format
        shape_data = {
            'x_start': shape.x0 - shape.width / 2,
            'x_stop': shape.x0 + shape.width / 2,
            'x_step': shape.width / (shape.x_count - 1),
            'x_number': shape.x_count,
            'y_start': shape.y0 - shape.height / 2,
            'y_stop': shape.y0 + shape.height / 2,
            'y_step': shape.height / (shape.y_count - 1),
            'y_number': shape.y_count
        }

        # Return the ssdtools.Shape object
        return cls(data=shape_data)

    def to_echo_api(self, api_client=None):
        """
        Method to send the Shape object to the Echo API.

        :param api_client:
        :return:
        """
        import echo_api_sdk

        # Create an instance of the GridsAPI class
        api = echo_api_sdk.GridsApi(api_client)

        # Create an echo_api_sdk.Grid
        grid = echo_api_sdk.Grid(
            x0=(self.x_start + self.x_stop) / 2,
            width=abs(self.x_stop - self.x_start),
            x_count=self.x_number,
            y0=(self.y_start - self.y_stop) / 2,
            height=abs(self.y_stop - self.y_start),
            y_count=self.y_number
        )

        # Get the echo_api.sdk.Grid
        return api.post_grid(grid=grid)


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
