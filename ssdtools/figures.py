import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import ticker
from cycler import cycler
from imageio import imread
from matplotlib import colors, colorbar, lines
from descartes import PolygonPatch
from geopandas import GeoDataFrame

from ssdtools import branding ###TODO Is dit nodig
from ssdtools.branding import default
from ssdtools.traffic import Traffic
from ssdtools.traffic import read_file



def soften_colormap_edge(colormap, transition_width=.25, alpha=1.):
    """
    Soften the colormap by applying a linear transition to zero at the front of the colormap.

    # # # # # # # # # < 1
    #   ----------- # < alpha
    #  /            #
    # /             #
    # # # # # # # # # < 0
    ^   ^           ^
    0   |           1
        |
    transition_width

    :param ColorMap colormap: the colormap to soften.
    :param float transition_width: the width (as percentage) of the transition, should be between 0 and 1.
    :param float alpha: the maximum alpha. Should have a value between 0 and 1.
    :rtype: ColorMap
    """
    # Get the colormap colors
    colormap_colors = colormap(np.arange(colormap.N))

    # Get the transition range
    transition_range = np.arange(int(colormap.N * transition_width))

    # Set alpha
    colormap_colors[:, -1] = alpha

    # Set the alpha in the transition range
    colormap_colors[transition_range, -1] *= np.linspace(0, 1, transition_range.shape[0])

    # Create new colormap
    return colors.ListedColormap(colormap_colors)


def soften_colormap_center(colormap, alpha=1.):
    """
    Soften the colormap by applying a linear transition from 1 to 0 at the first half and from 0 to 1 at the second
    half.

    # # # # # # < 1
    # \     / # < alpha
    #  \   /  #
    #   \ /   #
    # # # # # # < 0
    ^    ^    ^
    0   0.5   1

    :param ColorMap colormap: the colormap to soften.
    :param float alpha: the maximum alpha. Should have a value between 0 and 1.
    :rtype: ColorMap
    """
    # Get the colormap colors
    colormap_colors = colormap(np.arange(colormap.N))

    # Calculate the length of half the list
    n_2 = int(colormap.N / 2)

    # Set alpha
    colormap_colors[:, -1] = alpha

    # Set the alpha in the transition range
    colormap_colors[:n_2, -1] *= np.linspace(1, 0, n_2)
    colormap_colors[n_2:, -1] *= np.linspace(0, 1, colormap.N - n_2)

    # Create new colormap
    return colors.ListedColormap(colormap_colors)


class GridPlot(object):
    def __init__(self, grid, other_grid=None, title=None, branding=None, figsize=None, xlim=None, ylim=None,
                 background=None, schiphol_border=None, place_names=None, extent=None):

        # Create an ID
        self.id = str(pd.Timestamp.utcnow())

        # Set the grids
        self.grid = grid
        self.other = other_grid

        # Set the plotting options
        self.title = title
        self.branding = default if branding is None else branding
        self.figsize = (21 / 2.54, 21 / 2.54) if figsize is None else figsize
        self.xlim = (80000, 140000) if xlim is None else xlim
        self.ylim = (455000, 515000) if ylim is None else ylim
        self.background = 'lib/Schiphol_RD900dpi.png' if background is None else background
        self.schiphol_border = 'lib/2013-spl-luchtvaartterrein.shp' if schiphol_border is None else schiphol_border
        self.place_names = 'lib/plaatsnamen.csv' if place_names is None else place_names
        self.extent = [80000, 158750, 430000, 541375] if extent is None else extent

        # Create a new figure
        self.fig, self.ax = self.create_figure()

        # Create a placeholder for contour plots
        self.contour_plot = None

    def create_figure(self):
        """
        Initialize a new contour plot.

        :return: the figure and axis handles.
        """

        # Create a new figure with figsize
        fig = plt.figure(num=self.id, figsize=self.figsize)

        # Don't show axis
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Square X- and Y-axis
        ax.set_aspect('equal', 'box')

        # Zoom in
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        return fig, ax

    def add_background(self, background):
        """
        Add the background map to the figure.

        :param str|np.ndarray background: path to the background file or background image as NumPy array.
        """
        if isinstance(background, str):
            self.background = imread(background)
        self.ax.imshow(self.background, zorder=0, extent=self.extent)

    def add_place_names(self, place_names, color=(0.0, 0.3, 0.3)):
        """
        Add the place names to the map.

        :param str|pd.DataFrame place_names: path to the place name file or a data frame containing the place names with
        corresponding coordinates.
        :param tuple color: the color of the text.

        """
        if isinstance(place_names, str):
            self.place_names = pd.read_csv(place_names, comment='#')

        for index, row in self.place_names.iterrows():
            self.ax.annotate(row['name'], xy=(row['x'], row['y']), size=6, color=color, horizontalalignment='center',
                             verticalalignment='center')

    def add_terrain(self, terrain):
        """

        :param str|GeoDataFrame terrain: path to the terrain file or a Pandas DataFrame with a geometry column.
        """
        if isinstance(terrain, str):
            self.schiphol_border = GeoDataFrame.from_file(terrain)

        fc = 'none'
        ec = (.0, .0, .0)
        poly = self.schiphol_border['geometry'][0]
        patch = PolygonPatch(poly, fc=fc, ec=ec, lw=0.2, zorder=10)
        self.ax.add_patch(patch)

        # todo: Is dit noodzakelijk?
        im = self.ax.imshow(self.background, clip_path=patch, clip_on=True, zorder=9, extent=self.extent)
        im.set_clip_path(patch)

    def add_scale(self, ticks=None, color=(0.0, 0.3, 0.3)):
        """

        :param list(float) ticks: the ticks to use as scale, in km.
        :param tuple color: the color of the scale.
        """

        # Scale, with xpos and ypos as position in fig-units
        xpos = .95
        ypos = 0.04
        ticks = [0, 2, 5, 10] if ticks is None else ticks

        # Transform from data to fig
        l_data = ticks[-1] * 1000
        l_disp = self.ax.transData.transform((l_data, 0)) - self.ax.transData.transform((0, 0))
        inv_fig = self.fig.transFigure.inverted()
        l_fig = inv_fig.transform(l_disp)[0]

        ax2 = self.fig.add_axes([xpos - l_fig, ypos, l_fig, 0.01])

        # Transparent background
        ax2.patch.set_alpha(0)

        # Disable spines
        for loc in ['right', 'left', 'top']:
            ax2.spines[loc].set_color('none')

        # Remove Y-as
        ax2.yaxis.set_ticks([])

        # Add X-ticks
        ax2.xaxis.set_ticks(ticks)
        labels = ['{}'.format(tick) for tick in ticks]
        labels[-1] = '{} km'.format(ticks[-1])
        ax2.set_xticklabels(labels)

        # Format
        ax2.spines['bottom'].set_color(color)
        ax2.spines['bottom'].set_linewidth(0.5)
        ax2.tick_params(axis='x', labelsize=6, colors=color, length=4, direction='in', width=0.5)

    def add_contours(self, level, primary_color=None, secondary_color=None,label='create label',other_label='create other label',refine_factor=20):
        """
        Add a contour of the grid at the specified noise level. When a multigrid is provided, the bandwidth of the contour
        will be shown.

        :param float level: the noise level of the contour to plot.
        :param primary_color: color for the main contour.
        :param secondary_color: color for the secondary contours (only used for multigrids).
        :return:
        """

        # Select this plot as active figure
        self.select()

        # Refine the grid
        shape = self.grid.shape.copy().refine(refine_factor)
        
        # Extract the x and y coordinates
        x = shape.get_x_coordinates()
        y = shape.get_y_coordinates()

        # If the grid is a multigrid, all noise levels should be plotted.
        if isinstance(self.grid.data, list):

            # Get the various statistics of the data
            statistic_grids = self.grid.statistics()

            # Extract the data of interest
            mean_grid = statistic_grids['mean'].resize(shape)
            dhi_grid = statistic_grids['dhi'].resize(shape)
            dlo_grid = statistic_grids['dlo'].resize(shape)

            # Plot the contour area
            colormap = colors.ListedColormap([secondary_color])
            area_mask = np.logical_or(dhi_grid.data < level, dlo_grid.data > level)
            area_grid = np.ma.array(mean_grid.data, mask=area_mask)
            cp = plt.contourf(*np.meshgrid(x, y), area_grid, cmap=colormap, alpha=0.4)

            # Plot the contours of the statistics
            cs = self.ax.contour(x, y, mean_grid.data, levels=[level], colors=primary_color, linewidths=[1, 1])
            cs_hi = self.ax.contour(x, y, dhi_grid.data, levels=[level], colors=secondary_color, linewidths=[0.5, 0.5])
            cs_lo = self.ax.contour(x, y, dlo_grid.data, levels=[level], colors=secondary_color, linewidths=[0.5, 0.5])
            
            legend_elements = [Line2D([0], [0], color=default['kleuren']['schemergroen']),
                               Line2D([0], [0], color=default['kleuren']['schipholblauw'])]

            self.ax.legend(legend_elements, ['58 Lden', '48 Lden'], loc='upper left',fontsize=12)
            

            
            

        # The input is a single grid, so only a single contour should be plotted
        else:
            grid = self.grid.copy().resize(shape)
            cs1 = self.ax.contour(x, 
                                 y, 
                                 grid.data, 
                                 levels=[level], 
                                 colors=primary_color, 
                                 linewidths=[1, 1])
            
        if self.other is not None:
            shape_other = self.other.shape.copy().refine(refine_factor)
                    
            # Extract the x and y coordinates
            x = shape_other.get_x_coordinates()
            y = shape_other.get_y_coordinates()
            grid = self.other.copy().resize(shape_other)
            cs2 = self.ax.contour(x, 
                                 y, 
                                 grid.data, 
                                 levels=[level], 
                                 colors=secondary_color, 
                                 linewidths=[1, 1])
    
            h1,_ = cs1.legend_elements()
            h2,_ = cs2.legend_elements()
            self.ax.legend([h1[0], h2[0]], [label,other_label],loc='upper left',fontsize=12)
#        legend_elements = [
#                           Line2D([0], [0], color=default['kleuren']['schemerblauw']),
#                           Line2D([0], [0], color=default['kleuren']['schemergroen']),
#                           Line2D([0], [0], color=default['kleuren']['middagblauw']),
#                           Line2D([0], [0], color=default['kleuren']['schipholblauw']),
#                           Line2D([0], [0], color=default['kleuren']['middaglichtblauw']),
#                           Line2D([0], [0], color=default['kleuren']['wolkengrijs_1'])
#                           ]
#
#        self.ax.legend(legend_elements, ['1-4 vluchten', 
#                                         '5-9 vluchten', 
#                                         '10-49 vluchten', 
#                                         '50-99 vluchten', 
#                                         '100-199 vluchten',
#                                         '200+ vluchten'],
#                loc='upper left',fontsize=12,title='Aantal vluchten boven 60dB(A)')
        return 


    def add_individual_contours(self, level, primary_color=None, secondary_color=None, refine_factor=20):
        """
        Add a contour of the grid at the specified noise level. When a multigrid is provided, all contours of the
        individual grids will be shown.

        :param float level: the noise level of the contour to plot.
        :param primary_color: color for the main contour.
        :param secondary_color: color for the secondary contours (only used for multigrids).
        :return:
        """

        # Select this plot as active figure
        self.select()

        # Refine the grid
        grid = self.grid.copy().refine(refine_factor)

        # Extract the x and y coordinates
        x = grid.shape.get_x_coordinates()
        y = grid.shape.get_y_coordinates()

        # If the grid is a multigrid, all noise levels should be plotted.
        if isinstance(grid.data, list):

            # Get the various statistics of the data
            statistic_grids = self.grid.statistics()

            # Extract the data of interest
            mean_grid = statistic_grids['mean'].resize(grid.shape)
            dhi_grid = statistic_grids['dhi'].resize(grid.shape)
            dlo_grid = statistic_grids['dlo'].resize(grid.shape)

            # Plot all individual contours
            for year_data in grid.data:
                cs_year = self.ax.contour(x, y, year_data, levels=[level], colors=secondary_color, linewidths=[3, 3],
                                          alpha=0.1)

            # Plot the contours of the statistics
            cs = self.ax.contour(x, y, mean_grid.data, levels=[level], colors=primary_color, linewidths=[1, 1])
            cs_hi = self.ax.contour(x, y, dhi_grid.data, levels=[level], colors=secondary_color, linewidths=[0.5, 0.5])
            cs_lo = self.ax.contour(x, y, dlo_grid.data, levels=[level], colors=secondary_color, linewidths=[0.5, 0.5])

        # The input is a single grid, so only a single contour should be plotted
        else:
            cs = self.ax.contour(x, y, grid.data, levels=[level], colors=primary_color, linewidths=[1, 1])

        return cs

    def add_heatmap(self, colormap=matplotlib.cm.get_cmap('jet'), soften_colormap=True, alpha=0.4, refine=1, refine_factor=20, **kwargs):
        """
        Show a grid by creating a heatmap.

        :param ColorMap colormap: the colormap to apply.
        :param bool soften_colormap: soften the colormap by making the edge transparent.
        :param float alpha: the maximum alpha. Should have a value between 0 and 1.
        :param int refine: a multiplication factor for the number of additional layers to plot, most colormaps consist
        of 64 colors.
        :param kwargs: optional arguments for the underlying contourf function.
        :return:
        """

        # Select this plot as active figure
        self.select()

        # Refine the grid
        grid = self.grid.copy().refine(refine_factor)

        # Extract the x and y coordinates
        x = grid.shape.get_x_coordinates()
        y = grid.shape.get_y_coordinates()

        # Add the transparency to the colormap
        if soften_colormap:
            colormap = soften_colormap_edge(colormap, transition_width=.25, alpha=alpha)

        # Plot the contour area
        self.contour_plot = self.ax.contourf(*np.meshgrid(x, y), grid.data, levels=refine * colormap.N, cmap=colormap,
                                             **kwargs)

        return self.contour_plot

    def add_comparison_heatmap(self, other_grid, colormap=matplotlib.cm.get_cmap('RdYlGn'), soften_colormap=True,
                               alpha=1.0, method='energetic',positive_scale=False, refine_factor=20, **kwargs):
        """
        Compare two grids by creating a heatmap.

        :param Grid other_grid: the noise grid to compare.
        :param ColorMap colormap: the colormap to apply.
        :param bool soften_colormap: soften the colormap by making the center transparent.
        :param float alpha: the maximum alpha. Should have a value between 0 and 1.
        :param kwargs: optional arguments for the underlying contourf function.
        :return:
        """

        # Select this plot as active figure
        self.select()

        # Align the shape of the other grid to the original grid
        diff_grid = other_grid.copy().resize(self.grid.shape)

        #compute scaling (energetically scale)
        if self.grid.unit == 'Lden':
            threshold = 48
        elif self.grid.unit == 'Lnight': 
            threshold = 40      
        elif self.grid.unit == 'LAmax': 
            threshold = 75  
        elif self.grid.unit == 'NAxx':
            threshold =0
            
        scale                               = np.ones(diff_grid.data.shape)
        scale[diff_grid.data<threshold]     = 10**((diff_grid.data[diff_grid.data<threshold]-threshold)/10)
        
        # Subtract the original grid from the other grid
        diff_grid.data -= self.grid.data   
        
        if method =='energetic':
            diff_grid.data *= scale

        # Refine the grid
        diff_grid.refine(refine_factor)

        # Extract the x and y coordinates
        x = diff_grid.shape.get_x_coordinates()
        y = diff_grid.shape.get_y_coordinates()

        # Add the transparency to the colormap
        if soften_colormap:
            colormap = soften_colormap_center(colormap, alpha=alpha)
            
        if soften_colormap and positive_scale:
            colormap = soften_colormap_edge(colormap, transition_width=1,alpha=alpha)

        # Plot the contour area
        self.contour_plot = self.ax.contourf(*np.meshgrid(x, y), diff_grid.data, levels=colormap.N, cmap=colormap,
                                             **kwargs)

        return self.contour_plot
    
    def add_colorbar(self, contour_plot=None,cax_position=None):

        # Use the contour plot of this object if no contour plot is provided
        contour_plot = self.contour_plot if contour_plot is None else contour_plot

        # Create new axis for the colorbar in the top-right corner. The sequence is left, bottom, width and height.
        cax = self.fig.add_axes([0.8, 0.6, 0.05, 0.3]) if cax_position is None else self.fig.add_axes(cax_position)

        # Add the colorbar
        return colorbar.ColorbarBase(cax, cmap=contour_plot.get_cmap(), norm=colors.Normalize(*contour_plot.get_clim()))

    def select(self):
        plt.figure(self.id)
        plt.sca(self.ax)

    def save(self, *args, **kwargs):
        return self.fig.savefig(*args, **kwargs)

    def show(self):
        return self.fig.show()


def plot_season_traffic(distribution, column_colors=None):
    """
    A function to create a traffic per season plot. Can also be used to plot other grouped horizontal stacked bar
    charts.

    :param dict column_colors: a dict containing the colors for the columns.
    :param pd.DataFrame distribution: a dataframe containing the numbers to visualise. The dataframe should have a
    2-level multiindex, where the first level is the seasons and the second level is the type of operation. The columns
    are used as labels for the data.
    :return: figure and axes.
    """

    # Get the seasons
    seasons = distribution.index.get_level_values(0).unique()

    # Create a subplot for each season
    fig, ax = plt.subplots(len(seasons))
    plt.subplots_adjust(left=.25, hspace=0.0)
    fig.set_size_inches(30 / 2.54, 10 / 2.54)

    # Add the data to each subplot
    for i, season in enumerate(seasons):
        # Select the data
        season_data = distribution.loc[season]

        # Create a cumulative sum to get the indentation of the bars right
        season_data_cumsum = np.zeros(season_data.shape, dtype=int)
        season_data_cumsum[:, 1::] = np.cumsum(season_data.values[:, :-1], axis=1)

        # Add the column
        for j, column in enumerate(season_data.columns):
            if column_colors is not None and column in column_colors:
                color = column_colors[column]
            else:
                color = None

            ax[i].barh(season_data.index, season_data.values[:, j], left=season_data_cumsum[:, j], label=column,
                       color=color)

        ax[i].set_ylabel(season)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        # Add vertical grid lines
        ax[i].grid(axis='x')

        ax[i].set_xlim([0, distribution.sum(axis=1).max()])

    # Remove the top ticks
    for i in range(len(seasons) - 1):
        ax[i].axes.xaxis.set_ticklabels([])

    return fig, ax


def plot_aircraft_types(traffic_aggregate, ax=None, **kwargs):
    # Extract the weight class
    weight_class = pd.concat([traffic_aggregate.data['total'],
                              traffic_aggregate.data['d_ac_cat'].str.get(0).fillna(0).astype(int)], axis=1)
    weight_class.columns = ('total', 'weight_class')

    # Set the MTOW definitions
    mtow_def = pd.Series({
        0: '< 6',
        1: '6 - 40',
        2: '6 - 40',
        3: '40 - 60',
        4: '60 - 160',
        5: '60 - 160',
        6: '160 - 230',
        7: '230 - 300',
        8: '> 300',
        9: '> 300'
    }).reset_index(drop=False, name='MTOW').rename(columns={'index': 'weight_class'})

    # Add the MTOW definitions to the weight classes
    weight_class = pd.merge(weight_class, mtow_def, on='weight_class', how='left')

    # Calculate the fleet distribution by MTOW
    fleet = weight_class.groupby(['MTOW'])['total'].sum()
    fleet = fleet.reindex(mtow_def['MTOW'].unique())
    fleet = fleet / fleet.sum() * 100
    fleet = fleet.fillna(0)
    print (fleet)

    if ax is None:
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 4))

    # Add horizontal grid lines
    ax.grid(axis='y')

    # Create the bar plot
    ax.bar(range(fleet.shape[0]), fleet, **kwargs)

    # Set the x-ticks
    plt.xticks(range(fleet.shape[0]), fleet.index.tolist())

    # Format the y-ticks
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d %%'))

    # Set the x-label
    ax.set_xlabel('Maximum startgewicht in tonnen')

    # Get rid of box around graph
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    

    try:
        return fig, ax
    except NameError:
        return None, ax

class BracketPlot(object):
    def __init__(self, slond_colors=None, figsize=None, capacity_color='#cdbbce', takeoff_color='#4a8ab7',
                 landing_color='#fdbb4b'):
        # Create an ID
        self.id = str(pd.Timestamp.utcnow())

        # Set the plotting options
        self.figsize = (21 / 2.54, 21 / 2.54) if figsize is None else figsize
        self.capacity_color = capacity_color
        self.slond_colors = slond_colors if slond_colors is not None else {
            'S': '#cdbbce',
            'L': '#cdbbce',
            'O': '#8fbbd6',
            'N': '#d1e6bd',
            'D': '#f1b7b1'
        }
        self.landing_color = landing_color
        self.takeoff_color = takeoff_color

        # Create a new figure
        self.fig, self.ax = self.create_figure()

        # Create a placeholder for the capacity bracket
        self.capacity_bracket = None

    def create_figure(self):
        """
        Initialize a new bracket plot.

        :return: the figure and axis handles.
        """

        # Create a new figure with figsize
        fig, ax = plt.subplots(num=self.id, figsize=self.figsize)
        fig.set_size_inches(21 / 2.54, 9 / 2.54)

        return fig, ax

    def add_capacity_bracket(self, bracket=None):
        self.select()

        if bracket is None:
            bracket = self.capacity_bracket
        else:
            self.capacity_bracket = bracket

        # Get the x-coordinates for the brackets
        x = bracket.data.columns + .5

        # Add the SLOND colors
        if 'period' in bracket.data.index:
            color = bracket.data.loc['period', :].map(self.slond_colors).values
        else:
            color = self.capacity_color

        # Plot the takeoffs on top and the landings on the bottom
        self.ax.bar(x, -bracket.data.loc['L', :], width=0.92, color=color, edgecolor=None)
        self.ax.bar(x, bracket.data.loc['T', :], width=0.92, color=color, edgecolor=None)

    def add_traffic_bracket(self, bracket):
        self.select()

        x = bracket.data

        # Plot the takeoffs on top and the landings on the bottom
        self.ax.bar(x.columns + .5, -x.loc['L', :], width=0.5, facecolor=self.takeoff_color, edgecolor='#757575',
                    lw=0.25)
        self.ax.bar(x.columns + .5, x.loc['T', :], width=0.5, facecolor=self.landing_color, edgecolor='#757575',
                    lw=0.25)

        # Set the lines of the grid
        self.ax.set_axisbelow(True)
        self.ax.set_xticks(x.columns, minor=True)
        self.ax.xaxis.grid(which='minor')
        self.ax.yaxis.grid(which='major')

        # Set the ticks of the axes
        self.ax.axes.tick_params(axis='both', which='both', labelsize=6, labelrotation=0, labelcolor='#757575', pad=4)

        # Set the x-axis
        self.ax.set_xlim([0, 72])
        self.ax.set_xlabel('')  # size = 14)                    # verberg as-label
        self.ax.xaxis.set_tick_params(which='both', length=0)  # en geen tickmarks
        self.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1.5, 71, 3)))
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda u, v: '{:d}:00'.format(int(u // 3))))

        # Create a second x-axis to separate the hours
        ax2 = self.ax.twiny()
        ax2.spines["bottom"].set_position(('outward', 3))
        ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 72.5, 3) / 72))
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_tick_params(which='major', length=7, width=0.5, color='#757575')
        ax2.xaxis.set_ticklabels([])
        for side in ax2.spines:
            ax2.spines[side].set_color('none')

        # Set the y-axis
        self.ax.yaxis.set_tick_params(which='both', length=0)
        self.ax.yaxis.set_major_locator(ticker.ticker.FormatStrFormatter(5))
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda u, v: '{:1.0f}'.format(abs(u))))

    def add_takeoff_landing_label(self, takeoff_label='takeoffs', landing_label='landings'):
        self.select()

        ylim = self.ax.get_ylim()

        ymid = -ylim[0] / (-ylim[0] + ylim[1])
        for y, label in zip([ymid / 2, (1 + ymid) / 2], [landing_label, takeoff_label]):
            self.ax.text(-0.041, y,
                         label,
                         ha='right',
                         va='center',
                         rotation='vertical',
                         transform=self.ax.transAxes)
        for y1, y2 in zip([0, ymid], [ymid, 1]):
            line = Line2D([-0.035, -0.035], [0.02 + y1, y2 - 0.02],
                          lw=0.5, color='#757575', transform=self.ax.transAxes)
            line.set_clip_on(False)
            self.ax.add_line(line)

    def add_capacity_legend(self, label='runway capacity'):
        w = 1 / 72
        x = 0.72
        ys = [1.0492, 1.0353, 1.0492, 1.0561, 1.0457]
        heights = [4 * w, 4 * w, 3 * w, 2 * w, 3.5 * w]
        for s, y, h in zip(self.slond_colors, ys, heights):
            if 'period' in self.capacity_bracket.data.index:
                c = self.slond_colors[s]
                self.ax.text(x + w / 2, 1.07, s, fontsize=4, ha='center', va='center', transform=self.ax.transAxes)
            else:
                c = self.capacity_color

            rect = Rectangle(xy=(x, y), width=w, height=h, facecolor=c, edgecolor='white', linewidth=0.5, clip_on=False,
                             transform=self.ax.transAxes)
            self.ax.add_patch(rect)
            x += w

        self.ax.text(x + w / 2, 1.07, label, fontsize=6, ha='left', va='center', transform=self.ax.transAxes)

    def add_traffic_legend(self, label='traffic'):

        self.select()

        # Add traffic legend
        w = .5 / 72
        y = 1.07
        h1, h2 = [6 * w, 2 * w, 3 * w], [-2 * w, -4 * w, -7 * w]
        colors = [self.landing_color, self.takeoff_color]
        for c, heights in zip(colors, [h1, h2]):
            x = 0.92
            for h in heights:
                rect = Rectangle(xy=(x, y), width=w, height=h, facecolor=c, edgecolor='#757575', linewidth=0.5,
                                 clip_on=False, transform=self.ax.transAxes)
                self.ax.add_patch(rect)
                x += w

        self.ax.text(x + w / 2, 1.07, label, fontsize=6, ha='left', va='center', transform=self.ax.transAxes)

    def select(self):
        plt.figure(self.id)
        plt.sca(self.ax)

    def save(self, *args, **kwargs):
        return self.fig.savefig(*args, **kwargs)

    def show(self):
        return self.fig.show()

# -----------------------------------------------------------------------------
# Get colors
# -----------------------------------------------------------------------------
def get_cycler_color(index=0):
    ''' Cycler color o.b.v. de index
    '''
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][index]

# -----------------------------------------------------------------------------
# legenda
# -----------------------------------------------------------------------------
def update_legend_position(ax, target, fig=None):
    '''Update legend position, target, o.b.v. bbox.x1
    '''
    if fig is None:
        fig = plt.gcf()
    transf = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()

    bbox = ax.get_tightbbox(renderer).transformed(transf)
    diff = target-bbox.x1

    l, b, w, h = ax.get_position().bounds
    # corrigeer x-positie
    ax.set_position([l+diff, b, w, h])


def get_text_bbox(handle, ax=None, fig=None):
    '''Bounding box of text
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax = plt.gca()

    fig.canvas.draw()
    return handle.get_window_extent().inverse_transformed(ax.transData)

     
def plot_history(history,
                 history_kwargs = {'sheet_name': 'realisatie'},
                 x='jaar',
                 y='verkeer',
                 label=['handelsverkeer'],
                 xlabel='jaar',
                 ylabel='vliegtuigbewegingen',
                 xstep=1,
                 ystep=None,
                 ncol=None, ###Todo: via xParams
                 clip_on=False,
                 dpi=600,
                 fname='',                 
                 figsize=(8.27, 2.76),
                 **kwargs):
    """
    :param pd.DataFrame|str history: the historic dataset to visualise, should contain the specified x and y as columns.
    If it is a string then the file will be inported in a pd.DataFrame.
    :param dict history_kwargs: optional arguments for read_file
    :param int|str x: the column name of the data to visualise, defaults to 'jaar'.
    :param int|str y: the column name of the data to visualise, defaults to 'verkeer'.
    :param str label: label for the legend
    :param str xlabel: label for the x-axis
    :param str ylabel: label for the y-axis
    :param float|None xstep: step value for the x-axis
    :param float|None ystep: step value for the y-axis
    :param int ncol: number of columns in legend
    :param boolean clip_on: clipping on plot area, default False    
    :param int dpi: dpi for saving figure to file, default is 600
    :param str fname: Name for the file, default is '' and no fig will be saved
    :param set figsize: Figsize in inches, default (21/2.54, 7/2.54)
    :return: if fname='', return a Matplotlib figure and axes.
    """
    
    def NumberFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:,.0f}'.format(x).replace(',', '.')
    
    # Import history data in dataframe
    if isinstance(history, str):
        history = read_file(history, **history_kwargs)
    
    # Plot the history
    ax = history.plot(x=x,
                      y=y,
                      figsize=figsize,
                      clip_on=clip_on,
                      **kwargs)
    
    # margins
    plt.subplots_adjust(**branding.xParams['subplots_adjust'])

    # X-as
    if xlabel is not None:
        branding.set_xlabels(xlabel, ax=ax)
    else:
        ax.set_xlabel('') # verberg as-label

    ###TODO: Ed geef opmaak mee in de invoer van functie
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    if xstep is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))

    # Y-as
    if xlabel is not None:
        branding.set_ylabels(ylabel, ax=ax)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(NumberFormatter))
    if ystep is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))

    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **branding.xParams['legend'])

    # save figure
    fig = plt.gcf()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax

def plot_prediction(history, 
                    prediction=None,
                    stats=None,
                    history_kwargs = {'sheet_name': 'realisatie'},
                    x='jaar',
                    y='verkeer',
                    labels=['realisatie', 'prognose'],
                    xlabel='jaar',
                    ylabel='vliegtuigbewegingen',
                    xstep=1,
                    ystep=None,
                    ncol=None, ###Todo: via xParams
                    clip_on=False,                   
                    dpi=600,
                    fname='',                 
                    figsize=(8.27, 2.76),
                    **kwargs
                    ):

    """
    :param pd.DataFrame|str history: the historic dataset to visualise, should contain the specified x and y as columns.
    If it is a string then the file will be inported in a pd.DataFrame.
    :param pd.DataFrame prediction: the raw predicted values, should contain the specified column_name as the data and a
    x-column. Will be superseeded if stats are given
    :param pd.DataFrame stats: the min, max and mean predicted values and year as index.
    :param dict history_kwargs: optional arguments for read_file
    :param int|str x: the column name of the data to visualise, defaults to 'jaar'.
    :param int|str y: the column name of the data to visualise, defaults to 'verkeer'.
    :param str label: label for the legend
    :param str xlabel: label for the x-axis
    :param str ylabel: label for the y-axis
    :param float|None xstep: step value for the x-axis
    :param float|None ystep: step value for the y-axis
    :param int ncol: number of columns in legend
    :param boolean clip_on: clipping on plot area, default False    
    :param int dpi: dpi for saving figure to file, default is 600
    :param str fname: Name for the file, default is '' and no fig will be saved
    :param set figsize: Figsize in inches, default (21/2.54, 7/2.54)
    :return: if fname='', return a Matplotlib figure and axes.
    """

    # Import history data in dataframe
    if isinstance(history, str):
        history = read_file(history, **history_kwargs)

    # Plot the history
    fig, ax = plot_history(history=history,
                           history_kwargs=history_kwargs,
                           x=x,
                           y=y,
                           label=labels[0],
                           xlabel=xlabel,
                           ylabel=ylabel,
                           xstep=xstep,
                           ystep=ystep,
                           ncol=ncol, ###Todo: via xParams
                           clip_on=clip_on,
                           fname='',                 
                           figsize=figsize,
                           zorder=3, # plot on top
                           **kwargs)
    
    # Describe the prediction for each year
    if stats is None:
        stats = prediction.groupby(x)[y].describe()
    means = stats['mean']
    lo = stats['min']
    hi = stats['max']
    
    # Combine last point from history with prediction
    history = history.set_index(x)
    p_mean = history[y].tail(1).append(means)
    p_lo = history[y].tail(1).append(lo)
    p_hi = history[y].tail(1).append(hi)

    # Color the background of the prediction
    ax.fill_between(p_mean.index,
                    p_lo,
                    p_hi,
                    clip_on=clip_on,
                    **branding.xParams['prediction_fill'])
    
    # Plot the error bars
    e = ax.errorbar(means.index,
                    means,
                    yerr=[means-lo, hi-means],  # asymmetrisch interval
                    label=labels[1],
                    # clip_on=clip_on, #Bug in Matplotlib? Lijkt niet te werken
                    **branding.xParams['errorbar'])
    # clip_on?
    if not clip_on:
        for b in e[1]:
            b.set_clip_on(False)
    
    # Plot de lijnen voor het gemiddelde
    ax.plot(p_mean.index,
            p_mean,
            color=get_cycler_color(1),
            marker='None',
            clip_on=clip_on)
    
    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **branding.xParams['legend'])

    # save figure
    fig = plt.gcf()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax


def plot_prediction2(history, prediction, column_name='data', prediction_errorbar_kwargs=None,
                    prediction_fill_kwargs=None, history_plot_kwargs=None,doc29_factor=None):
    """

    :param pd.DataFrame history: the historic dataset to visualise, should contain the specified column_name as the data
    and a 'year' column.
    :param pd.DataFrame prediction: the predicted values, should contain the specified column_name as the data and a
    'year' column.
    :param int|str column_name: the column name of the data to visualise, defaults to 'data'.
    :param dict history_plot_kwargs: argument arguments to overwrite the settings used for visualising the historic data.
    :param dict prediction_errorbar_kwargs: arguments to overwrite the settings used for visualising the errorbars of
    the prediction.
    :param dict prediction_fill_kwargs: arguments to overwrite the settings used for visualising the filled area
    of the prediction.
    :return: a Matplotlib figure and axes.
    """
    # Apply the custom history plot style if provided
    history_style = {'marker': 'o', 'markeredgewidth': 2, 'fillstyle': 'none', 'label': 'history',
                     'color': '#141251'}
    if history_plot_kwargs is not None:
        history_style.update(history_plot_kwargs)

    # Apply the custom prediction errobar style if provided
#    prediction_style = {'marker': '_', 'capsize': 4, 'ecolor': '#9491AA', 'markeredgewidth': 4,
#                        'markeredgecolor': '#9491AA', 'fillstyle': 'none', 'color': '#1B60DB', 'label': 'prediction'}
    if prediction_errorbar_kwargs is not None:
        prediction_style.update(prediction_errorbar_kwargs)

    # Apply the custom prediction fill_between style if provided
    prediction_fill = {'color': '#027E9B', 'alpha': 0.3}
    if prediction_fill_kwargs is not None:
        prediction_fill.update(prediction_fill_kwargs)


    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot the history
    plt.plot(history['years'], history[column_name], **history_style)
    
    # Describe the prediction for each year
    statistics = prediction[column_name]/doc29_factor

    # Plot the prediction
    plt.errorbar(history['years'].tail(1).tolist() + [prediction['years'].max()],
                 history[column_name].tail(1).tolist() + [statistics[1]],
                 yerr=[[0] + [(statistics[1]- statistics[0])],
                       [0] + [(statistics[2] - statistics[1])]],
                 **prediction_style)
    
    # Color the background of the prediction
    plt.fill_between(history['years'].tail(1).tolist() + [prediction['years'].max()],
                     history[column_name].tail(1).tolist() + [statistics[0]],
                     history[column_name].tail(1).tolist() + [statistics[2]],
                     **prediction_fill)


    # Set the xticks
    ax.set_xticks(np.arange(history['years'].min(), prediction['years'].max() + 1, 1))

    # Add horizontal grid lines
    ax.grid(axis='y')
    ax.set_ylim(bottom=0)
    # Add a legend
    plt.legend(ncol=2, bbox_to_anchor=(0.9, 1.15))
    
    if doc29_factor:
        ax.set_ylabel('NRM', color='k')  
    
    
        scale_factor=doc29_factor
        
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.plot(history['years'],scale_factor*history[column_name], alpha=0)
        
        ax2.set_ylabel('Doc. 29')  # we already handled the x-label with ax1
#        ax2.tick_params(axis='y', labelcolor='k')
        ax2.set_ylim(bottom=0)
    
    return fig, ax


def plot_windrose(windrose):
    # Get the directions
    directions = windrose.index.get_level_values(0).unique()
    directions = directions[directions > 0]

    # Get the directions
    speeds = windrose.index.get_level_values(1).unique()
    speeds = speeds[speeds > 0]

    # Fill zero values, necessary for bar stacking
    idx = pd.MultiIndex.from_product([directions, speeds])
    data = windrose.reindex(idx)

    # Calculate the angles
    theta = np.deg2rad(directions)

    # Calculate the percentages and reshape the data
    data_x = (data / windrose.sum() * 100).reset_index().pivot(index='level_0', columns='level_1', values='STN')
    bottom = np.zeros(data_x.shape) + 5
    bottom[:, 1:] += data_x.iloc[:, :-1].cumsum(axis=1).values
    bottom_x = pd.DataFrame(bottom, index=data_x.index, columns=data_x.columns)

    # Create a polar axis system
    ax = plt.subplot(111, projection='polar', facecolor='0.9')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 30)
    ax.set_yticks([5, 15, 25, 35])
    ax.set_yticklabels(['', '10 %', '20 %', '30 %'])
    ax.set_xticks(theta)
    ax.set_xticklabels(['', '', 'O', '', '', 'Z', '', '', 'W', '', '', 'N'])

    for max_speed in data_x.columns:

        # Calculate the minimum speed for this maximum speed
        min_speed = max_speed - 5

        # Set the label
        if max_speed != data_x.columns.max():
            label = '{}-{} kts'.format(min_speed, max_speed)
        else:
            label = '>{} kts'.format(min_speed)

        # Remove the nan values to avoid runtime warnings
        theta_s = theta[~bottom_x[max_speed].isna()]
        data_s = data_x.loc[~bottom_x[max_speed].isna(), max_speed]
        bottom_s = bottom_x.loc[~bottom_x[max_speed].isna(), max_speed]

        # Calculate the width of the bar
        bar_width = 2 * np.pi / 12 * (0.15 + max_speed / 50)

        # Plot the bars
        p1 = ax.bar(theta_s, data_s, bottom=bottom_s, width=bar_width, label=label)

    ax.bar(0, 5, width=2 * np.pi, label='other')
    plt.text(0, 0, '{:.1f} %'.format(windrose.loc[0, 0] / windrose.sum() * 100), horizontalalignment='center',
             verticalalignment='center')
    ax.legend(loc=4)
    plt.show()


def plot_runway_usage(traffic,
                      labels,
                      den=('D','E','N'),
                      n=7,
                      runways=None,
                      ylabel='vliegtuigbewegingen',
                      ylim=[0,110000],
                      dy=10000,
                      reftraffic=1,
                      numbers=False,
                      fname=None,
                      dpi=600):
    '''Plot het baangebruik'''

    def NumberFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:,.0f}'.format(x).replace(',', '.')
    def GetVal(var, i):
        'Scalar of een list'
        if isinstance(var, list):
            i = min(i, len(var)-1) # hergebruik van de laatste waarde
            return var[i]
        else:
            return var

    # kopie van plotformat
    MarkerWidth =  [0.36, 0.36, 0.2, 0.2, 0.1]  # voor 1, 2 en >2 traffics
    MarkerHeight = [0.12, 0.12, 0.1, 0.1, 0.08]
    BarWidth = [0.16, 0.16, 0.06, 0.06, 0.04]
    BarGap = [0, 0, 0.05, 0.05, 0.04]

    # converteer naar list
    if not isinstance(traffic, list): traffic = [traffic]
    if not isinstance(labels, list): labels = [labels]

    # X-positie van de bars
    x = np.arange(n)

    ntrf = len(traffic)
    i = ntrf - 1
    w = GetVal(BarWidth, i) * n/7  # normaliseer voor de aslengte
    g = GetVal(BarGap, i)          # of /ntrf?

    dx = [(w+g)*(x - 0.5*(ntrf-1)) for x in range(ntrf)]

    # markers en staafjes
    marker_height = (GetVal(MarkerHeight, i) * (ylim[-1] - ylim[0]) / 10)
    mw = GetVal(MarkerWidth, i) * n/7
    dxm = list(dx)

    # clip marker
    if ntrf == 2:
        mw = (mw + w)/2
        dxm[0] = dx[0] - (mw-w)/2
        dxm[1] = dx[1] + (mw-w)/2
    elif ntrf > 2:
        mw = min(mw, w+g)

    # twee aansluitende subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(21/2.54, 10/2.54)

    # margins
    fig.subplots_adjust(bottom=0.18, wspace=0)

    # verwerken traffics
    for i, trf in enumerate(traffic):
        
        # Read traffic
        if isinstance(trf, str):
            trf = Traffic.read_daisy_meteoyear_file(trf)

        # Get the runway statistics
        trf_stats = trf.get_runway_usage_statistics('|'.join(den)).reset_index()

        # sorteer
        if 'key' not in trf_stats.columns:
            trf_stats['key'] = trf_stats['d_lt'] + trf_stats['d_runway']

        if runways is not None:
            # tweede traffic in dezelfde volgorde
            keys = [k + r for k in runways for r in runways[k]]    # keys: combinatie van lt en runway
            sorterIndex = dict(zip(keys, range(len(keys))))        # plak een volgnummer aan de keys
            trf_stats['order'] = trf_stats['key'].map(sorterIndex) # soteerindex toevoegen
            trf_stats = trf_stats.sort_values(by=['order'])        # sorteer dataframe
        else:
            trf_stats = trf_stats.sort_values(by=['d_lt', 'mean'], ascending=False)
            runways = {'L': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'L'],
                       'T': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'T']}

        # maak de plot
        for lt, xlabel, ax in zip(['T', 'L'], ['starts', 'landingen'], [ax1, ax2]):

            # selecteer L of T
            trf2 = trf_stats.loc[trf_stats['d_lt'] == lt]
            trf2 = trf2.head(n) # gaat alleen goed als er ook echt n-runways zijn

            # staafjes
            bar_height = trf2['max'] - trf2['min']
            if i == reftraffic:
                c = 1
            else:
                c = 0

            ax.bar(x+dx[i],
                   height=bar_height.values,
                   bottom=trf2['min'].values,
                   width=w,
                   color=get_cycler_color(c))

            # gemiddelde
            ax.bar(x+dxm[i],
                    height=marker_height,
                    bottom=trf2['mean'].values-marker_height/2,
                    width=mw,
                    color=get_cycler_color(c))

            # opmaak, alleen de eerste keer
            if i == 0:
                # geen vertikale gridlines
                ax.grid(which='major', axis='x', b=False)

                # X-as
                ax.margins(x=0.02)
                ax.set_xticks(x)
                ax.set_xticklabels(trf2['d_runway'])
                branding.set_xlabels(xlabel, gap=0.02, ax=ax)

                # Y-as
                ax.set_ylim(ylim)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=dy))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(NumberFormatter))
                if ax==ax1:
                    branding.set_ylabels(ylabel, ax=ax)


    # scheidingslijntje tussen subplots
    for ax, side, i in [[ax1, 'right', 1], [ax2, 'left', 0]]:
        ax.spines[side].set_color('none')
        ax.axvline(ax.get_xlim()[i],
                   marker='',
                   color='white',
                   linewidth=2,
                   zorder=0)

    # nummertjes
    if numbers:
        xp = []
        yp = []
        for i, p in enumerate(ax1.patches):
            if not i%(n*2):
                xp.append(p.get_x() + p.get_width()/2)
                yp.append(p.get_y())
        for i, x in enumerate(xp):
            ax1.text(x, 0.95 * min(yp), str(i),
                     ha='center', va='top', fontsize=3)

    # legenda
    if ntrf == 1:
        w *= 0.8      # legenda op 80%
        g *= 0.8
        mw *= 0.8
        dx = [0]      # x voor de staafjes
        dxm = [0]     # x voor de markers
        plot_matrix = [(0, 0.30, 0.05, 0.2, 'left')]        
    elif ntrf == 2:
        w *= 0.8                # legenda op 80%
        g *= 0.8
        mw *= 0.8
        dx = [-w/2, w/2]        # x voor de staafjes
        dxm = [-mw/2, mw/2]     # x voor de markers
        plot_matrix = [(0, 0.3, 0.05, -0.2, 'right'),
                       (1, 0.4, 0.15, 0.2, 'left')]
    else:
        dx = [(w+g)* x for x in range(4)]
        dxm = dx
        reftraffic=3
        plot_matrix = [(0, 0.30, 0.05, 0.15, None),
                       (1, 0.25, 0.00, 0.15, None),
                       (2, 0.35, 0.10, 0.15, 'left'),
                       (3, 0.30, 0.05, 0.15, 'left')]

    # plotgebied
    ax0 = fig.add_axes([0, 0.89, 0.025, 0.1])

    # geen assen
    ax0.axis('off')

    # genormaliseerde asses
    ax0.set_xlim(0, 0.5*n/7)
    ax0.set_ylim(0, 1)

    for i, yi, bottom, xt, ha in plot_matrix:
        if i == reftraffic:
            c = 1
        else:
            c = 0
        ax0.bar(dx[i], height=0.5, bottom=bottom,
                width=w,
                color=get_cycler_color(c),
                clip_on=False)
        ax0.bar(dxm[i], height=0.05, bottom=yi,
                width=mw,
                color=get_cycler_color(c),
                clip_on=False)
        if ha:
            t = ax0.text(dx[i]+xt , 0.50, labels[c],
                         horizontalalignment=ha,
                         verticalalignment='top')
        if i == 2:
            # maak ruimte voor text label
            bbox = get_text_bbox(t)
            dx[3] += bbox.x1


    # Check uitlijning van de legend
    update_legend_position(ax0, target=0.9-0.01)

    # save figure
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, (ax1, ax2)
