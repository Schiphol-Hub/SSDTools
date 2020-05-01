import datetime
from os.path import splitext

import xlrd
import numpy as np
import pandas as pd
import math

def read_file (filename, sep='\t', **kwargs):
    '''Importeer xls-, xlsx- of tekstbestand in een dataframe,
       als filename geen sting is dan is het waarschijnlijk al een dataframe'''
    if isinstance(filename, str):
        _, ext = splitext(filename)
        if ext in ['.xls', '.xlsx']:
            return pd.read_excel(filename, **kwargs)
        else:
            return pd.read_csv(filename, sep=sep, **kwargs)
    else:
         return filename

def emission_model(trf,
                   ET,
                   TIM,
                   ACtypes,
                   ac_cat,
                   f_1ipv2=0,
                   f_3ipv4=0,
                   f_2ipv3=0,
                   f_APU400hz=0,
                   f_APU=0.75,
                   reference_traffic=None,
                   new_engine=None):
    """
    Compute the emissionsoutput based on a daisy aggregate traffic

    TODO: beschrijf invoer en result
    """
    def mergeAndReplace(self,df1,df2,merge_l,merge_r,rep_l,rep_r):
        """
        Replaces values in a dataframe. Used for the emissiemodel code
        """
        
        df = df1.copy()
        
        # MERGE
        df = df.merge(df2, 
                      left_on=merge_l,
                      right_on=merge_r,
                      how='left')
    
        # copy values
        df.loc[df[rep_l].isnull(),rep_l] = df.loc[~df[rep_r].isnull(),rep_r]
        
        # drop columns
        df = df.drop(columns=rep_r)
    
        return df


    # Fill missing columns with NaN
    if 'engine_type' not in trf: trf['engine_type'] = np.nan
    if 'mtow' not in trf: trf['mtow'] = np.nan


    # ICAO or IATA type
    if len(trf.loc[0,'ac_type'])==3:
        aircraft_type = 'iata_aircraft'
    else:
        aircraft_type = 'icao_aircraft'

 
    # Add info from aircraft categories
    ac_cat = ac_cat.loc[:,['iata_aircraft','icao_aircraft','mtow', 'motor']]

    # drop duplicate codes
    ###TODO: eerste wordt bewaard, alternatief max, min, mean etc.
    ac_cat = ac_cat.drop_duplicates(subset=[aircraft_type])
    
    # mtow wordt door de merge mtow_x en mtow_y
    trf = trf.merge(ac_cat,
                    left_on='ac_type',
                    right_on=aircraft_type,
                    how='left')

    # Add engine if missing in trf
    trf['engine_type'] = trf['engine_type'].fillna(trf['motor'])
    trf = trf.drop(columns='motor')

    # Add mtow if missing in trf
    trf['mtow'] = trf['mtow_x'].fillna(trf['mtow_y'])
    trf = trf.drop(columns=['mtow_x', 'mtow_y'])
        
    # convert to LTO's
    trf['LTO'] = trf['total']/2            

    #%% traffic vanuit TAF verrijken met motornamen uit TIS
    ###TODO: Alleen als iata_codes worden gebruikt?
    if reference_traffic and aircraft_type == 'iata_aircraft':             
        # drop duplicates, but keep rows with most occurences
        reference_traffic = reference_traffic.sort_values(by='total',ascending=False).drop_duplicates(subset=['ac_type'])
        
        # add motornamen to traffic
        trf = trf.merge(reference_traffic, 
                        left_on='icao_aircraft',
                        right_on='ac_type',
                        how='left')

    ###TODO: Alleen als iata_codes worden gebruikt?
    if new_engine and aircraft_type == 'iata_aircraft':  
        # if empty columns --> new engine type.
        if any(trf['engine_type'].isnull()):
            # check if new engines can be added
            trf = mergeAndReplace(trf, new_engine,
                                       'icao_aircraft',
                                       'icao_aircraft',
                                       'engine_type',
                                       'engine_type')
            
            for engine,ac in zip(trf['engine_type'],trf['icao_aircraft']):
                if pd.isnull(engine):
                    print('Add missing engine to the following ICAO type: '+ str(ac))

        
    ###TODO: Echt alleen bij ICAO-codes?
    if aircraft_type == 'icao_aircraft':             
        # check for nans
        for aircraft in trf.loc[trf['icao_aircraft'].isnull(),'ac_type']:
            print('WARNING: missing aircraft type in aircraft categories table: '+ aircraft)
            total = trf.loc[trf['ac_type']==aircraft,'total'].sum()
            print(f'WARNING: will delete {total:.0f} flight(s)')
         
        # compute correction factor missing mtow
        t = trf['total'].sum()
        t_noMTOW = trf.loc[trf['mtow'].eq(0),'total'].sum()
        cfmtow = t/(t-t_noMTOW)
        print('Correction factor for missing mtow in ac cat table =' + str(round(cfmtow,4)))

        # compute correction factor missing ac cat
        t_noACCAT = trf.loc[trf['icao_aircraft'].isnull(),'total'].sum()
        cf = t/(t-t_noACCAT)
        print('Correction factor for missing aircraft in ac cat table =' + str(round(cf,4)))
        # drop columns with aircraft that don't exist in the database 
        trf = trf.dropna(subset=['icao_aircraft'])
        
    else:
        # correctionfactor for missing data
        cf = 1      # missing engine?
        cfmtow = 1  # missing mtow


    # print total MTOW
    mtow = sum(trf['total']*trf['mtow'])/1000
    print(f'\nTotal MTOW = {mtow:.1f} ton')
                                    
    # Add CO2 as pollutant
    modes = ['approach','idle','takeoff','climbout']
    for mode in modes:
        ET['co2_'+mode] = 3150
        
    modes = ['noload','power','airco','jetstart']
    for mode in modes:
        ACtypes['co2_'+mode] = 3150
   
    #%% Add info from engine types

    # expand table
    ET['_type'] = ET['type']
    for i, typ in enumerate(ET['type'].str.split(';')):
        if len(typ)>1:
            for subtype in typ:
                a = ET.iloc[i,:].copy()
                a['_type'] = subtype
                ET = ET.append(a)
    ET = ET.reset_index()
            
    # make case insensitive and trim trailing space
    trf['engine_type'] = trf['engine_type'].str.lower().str.strip()
    ET['_type'] = ET['_type'].str.lower().str.strip()
    
    # drop duplicates
    dups = ET['_type'].duplicated()
    if dups.any():
        for engine in ET.loc[dups, '_type'].unique():
            print('WARNING: duplicate engine types in engine table', engine)
        ET = ET.drop_duplicates(['_type'])
        
    # first merge on default types
    trf = trf.merge(ET,
                    left_on='engine_type',
                    right_on='_type',
                    how='left')
    
    # check for nans
    for engine in trf.loc[trf['_type'].isnull(),'engine_type']:
        print('WARNING: missing engine type table: '+ engine)
    
    #%% Add info from aircraft types
    trf = trf.merge(ACtypes,
                    left_on='icao_aircraft',
                    right_on='icao',
                    how='left')
    
    # check for missing engines and find replacement type
    trf.loc[(trf['_type'].isnull()) & (trf['mtow'] >5.7),'engine_type']= 'rb211-524b series package 1'
    trf.loc[(trf['_type'].isnull()) & (trf['mtow'] <5.7) & (trf['tim'] == 'TP') ,'engine_type']= '<5700 tp'
    trf.loc[(trf['_type'].isnull()) & (trf['mtow'] <5.7) & (trf['tim'] == 'P') ,'engine_type']= '< 5700 p'
    
    # drop columns and remerge
    trf    = trf.loc[:,['iata_aircraft','icao_aircraft','mtow','LTO','engine_type']]
    trf    = trf.merge(ET,
                       left_on='engine_type',
                       right_on='_type',
                       how='left')
    

    # check for nans
    for engine in trf.loc[trf['_type'].isnull(),'engine_type']:
        print('WARNING:\nAfter replacing, missing engine type table: '+ engine)
        # check for nans
        total = trf.loc[trf['engine_type']==engine,'LTO']
        print('Will delete: '+ str(total.iloc[0]*2) + ' flight(s)')

    # drop flights that don't exist in the database    
    trf = trf.dropna(subset=['_type'])
    
    #%% Add info from aircraft types
    trf = trf.merge(ACtypes,
                    left_on='icao_aircraft',
                    right_on='icao',
                    how='left')
    
    # check for nans
    for aircraft in trf.loc[trf['icao'].isnull(),'icao_aircraft']:
        print('WARNING: after replacing, missing engine type table: '+ aircraft)
        
    #%% Add info from TIM times
    trf = trf.merge(TIM,
                    left_on='tim',
                    right_on='code',
                    how='left')
    
    # TIM correction
    no = [2,3,4]
    correction = [f_1ipv2,f_2ipv3,f_3ipv4]
    for n,c in zip(no,correction):
        ids = (trf['engines']==n)
        trf.loc[ids,'idle'] = trf.loc[ids,'idle']-c*(trf.loc[ids,'idle']/2-3*60)*1/n
    
    
    #%% now compute emissies
    stoffen = ['co','nox','vos','so2','pm10','co2']
    modes = ['approach','idle','takeoff','climbout']
    
    d = {'Stof': stoffen}
    output = pd.DataFrame(data=d).set_index('Stof')
    
    for stof in stoffen:
        
        #%% LTO
        trf[stof+'_lto'] = 0
        for mode in modes:
            # compute uitstoot per stof, per mode
            trf[stof+'_lto_'+mode] = trf['fuel_'+mode]*trf[stof+'_'+mode]*trf[mode]

            # sommeer over de modes
            trf[stof+'_lto'] = trf[stof+'_lto']+trf[stof+'_lto_'+mode]
        
        # vermenigvuldig met het aantal motoren en LTO's
        trf[stof+'_lto'] = trf['LTO']*trf[stof+'_lto']*trf['engines']
        
        # sommeer over alle vluchten
        output.loc[stof,'LTO [t]'] = round(sum(trf[stof+'_lto'])/1000000 ,3)      
    
        #%% APU
    
        # APU +400 HZ
        trf[stof+'_APU400hz'] = (trf[stof+'_noload']*trf['fuel_noload']+
                                 0.5*trf[stof+'_airco']*trf['fuel_airco']+
                                 trf[stof+'_jetstart']*trf['fuel_jetstart'])
        # sommeer over alle vluchten
        output.loc[stof,'APU400hz [t]'] = round(f_APU400hz*sum(trf['LTO']*trf[stof+'_APU400hz'])/1000000,3)
        
        # APU
        trf[stof+'_APU'] = (trf[stof+'_noload']*trf['fuel_noload']+
                            trf[stof+'_power']*trf['fuel_power']+
                            0.5*trf[stof+'_airco']*trf['fuel_airco']+ 
                            trf[stof+'_jetstart']*trf['fuel_jetstart'])
        # sommeer over alle vluchten
        output.loc[stof,'APU [t]'] = round(f_APU*sum(trf['LTO']*trf[stof+'_APU'])/1000000,3)
    
        #%% total
    
        output.loc[stof,'Totaal [t]'] = output.loc[stof,'LTO [t]']+output.loc[stof,'APU400hz [t]']+output.loc[stof,'APU [t]']
        
        # correction factor based on missing aircraft in ac cat table
        output.loc[stof,'Totaal [t]'] *= cf
        
        output.loc[stof,'Totaal relatief [g/ton (MTOW)]']  = round(output.loc[stof,'Totaal [t]']*1000000/sum(trf['LTO']*trf['mtow']*2),3)  
        
        # correction factor based on missing mtow in ac cat table
        output.loc[stof,'Totaal relatief [g/ton (MTOW)]'] *= cfmtow
        
    # add total fuel
    output = output.append((output.loc['co2',:]/3.15).rename('fuel'))

    # drop temp column '_type'
    ET = ET.drop(columns=['_type'])
    trf = trf.drop(columns=['_type'])
    
    return trf, output.reset_index()


class Traffic(object):
    def __init__(self, data=None, date_column=None, class_column=None, id_column=None, den_column='DEN',
                 denem_column='DENEM', procedure_column='procedure', altitude_column='altitude',
                 weight_column='weight',aircraft_column='C_ac_type', engine_column='C_engine_type'):
        """

        :param pd.DataFrame data: traffic data
        """

        self.data = data
        self.date_column = date_column
        self.class_column = class_column
        self.id_column = id_column
        self.den_column = den_column
        self.denem_column = denem_column
        self.procedure_column = procedure_column
        self.altitude_column = altitude_column
        self.weight_column = weight_column
        self.aircraft_column=aircraft_column
        self.engine_column=engine_column

    @classmethod
    def read_daisy_phase_file(cls, path, **kwargs):
        """
        A method to read daisy phase files.
        E.g. summer seasons, winter seasons and maintenance periods.

        :param str path: path to the file.
        :return: daisy phase aggregate of traffic.
        :rtype: TrafficAggregate
        """
        return TrafficAggregate(data=read_file(path, **kwargs), aggregate_type='daisy.phase')

    @classmethod
    def read_daisy_meteoyear_file(cls, path, **kwargs):
        """
        A method to read daisy meteoyear files.

        :param str path: path to the file.
        :return: daisy meteoyear aggregate of traffic.
        :rtype: TrafficAggregate
        """
        ###TODO: Test, is index_col=None ndig?
        return TrafficAggregate(data=read_file(path, **kwargs), aggregate_type='daisy.meteoyear')

    @classmethod
    def read_daisy_runway_combination_file(cls, path, **kwargs):
        """
        A method to read daisy runway combination files.

        :param str path: path to the file.
        :return: daisy runway combination aggregate of traffic.
        :rtype: TrafficAggregate
        """

        # Read the file as DataFrame
        data_frame = read_file(path, **kwargs)

        # todo: Split the runway combination (d_combination)

        # Return the traffic object
        return TrafficAggregate(data=data_frame, aggregate_type='daisy.runway_combination')

    @classmethod
    def read_daisy_mean_file(cls, path, **kwargs):
        """
        A method to read daisy mean files.

        :param str path: path to the file.
        :return: daisy mean aggregate of traffic.
        :rtype: TrafficAggregate
        """

        return TrafficAggregate(data=read_file(path, **kwargs), aggregate_type='daisy.mean')
    
    @classmethod
    def read_daisy_HG_file(cls, path, **kwargs):
        """
        A method to read daisy mean files for HG computation.

        :param str path: path to the file.
        :return: daisy mean aggregate of traffic.
        :rtype: TrafficAggregate
        """

        return TrafficAggregate(data=read_file(path, **kwargs), aggregate_type='daisy.HG')

    @classmethod
    def read_daisy_weekday_file(cls, path, **kwargs):
        """
        A method to read daisy weekday files.

        :param str path: path to the file.
        :return: daisy weekday aggregate of traffic.
        :rtype: TrafficAggregate
        """

        return TrafficAggregate(data=read_file(path, **kwargs), aggregate_type='daisy.weekday')

    @classmethod
    def read_taf_file(cls, path, **kwargs):
        """
        A method to read sir TAF files.

        :param str path: path to the file.
        :return: sir TAF aggregate of traffic.
        :rtype: TrafficAggregate
        """

        # Read the csv file
        data = read_file(path, **kwargs)

        # Replace A (arrival) and D (departure) by L (landing) and T (takeoff)
        data['d_lt'] = data['d_lt'].str.replace(r'^A$', 'L').str.replace(r'^D$', 'T')

        # Return the traffic aggregate
        return TrafficAggregate(data=data, aggregate_type='taf.sir')

    @classmethod
    def read_casper_file(cls, path, sep=',', **kwargs):

        # Parse the file
        data = read_file(path, sep=sep, **kwargs)

        # Convert the dates
        data['C_actual'] = pd.to_datetime(data['C_actual'], format='%Y-%m-%d %H:%M:%S')

        return cls(data, date_column='C_actual', class_column='C_Klasse', id_column='C_id')

    @classmethod
    def read_nlr_file(cls, path, **kwargs):

        # Open the .xlsx file (this might take a while, but this is the only way to open large .xlsx files...)
        workbook = xlrd.open_workbook(path, on_demand=True)

        # Select the first worksheet
        worksheet = workbook.sheet_by_index(0)

        # Extract the data, column by column, with the first row as the column name
        data = {}
        for col in range(worksheet.ncols):
            data[worksheet.cell_value(0, col)] = worksheet.col_values(col, 1)

        # Put the data in a DataFrame
        data_frame = pd.DataFrame(data)
        
        ###TODO Test of bovenstaande code kan worden vervangen door
        # data_frame = read_file(path, **kwargs)

        # Create a datetime column
        data_frame['timestamp'] = pd.to_datetime(data_frame['Datum'] + data_frame['Tijd (LT)'], unit='D',
                                                 origin=pd.Timestamp(1900, 1, 1))

        # Return the traffic object
        return cls(data_frame, date_column='timestamp', class_column='Klasse', id_column='FlightId')

    def add_season(self, date_column=None):
        """
        Add the season (summer or winter) to the traffic, based on the date of the entry.

        :param date_column: the column with the date to use.
        """

        # Select the date column to use
        date_column = date_column if date_column is not None else self.date_column

        # Get the years
        years = self.data[date_column].dt.year.unique()

        # Put the season by default on winter
        self.data['season'] = 'winter'

        # Check for each year if the season should be summer
        for year in years:
            # Get the start dates for the two seasons and check which dates match the summer season
            after_start_summer = self.data[date_column] >= start_summer_season(year)
            before_start_winter = self.data[date_column] < start_winter_season(year)

            # Update the season for the matches
            self.data.at[np.logical_and(after_start_summer, before_start_winter), 'season'] = 'summer'

    def add_landing_takeoff(self, class_column=None):
        """
        Add takeoff/landing to the traffic, based on the four-digit traffic class.

        :param class_column: the column with the four-digit class to use.
        """

        # Select the class column to use
        class_column = class_column if class_column is not None else self.class_column

        # Add a departure/arrival column
        self.data['LT'] = None

        # Make sure the the class column is a string
        self.data.at[(self.data[class_column] >= 0) & (self.data[class_column] < 1000), 'LT'] = 'T'
        self.data.at[(self.data[class_column] >= 1000) & (self.data[class_column] < 2000), 'LT'] = 'L'

    def add_procedure(self):
        """
        Add the procedure to the traffic, based on the four-digit traffic class.
        """

        # Add a procedure, altitude and weight column
        self.data[self.procedure_column] = None
        self.data[self.altitude_column] = None
        self.data[self.weight_column] = None

        # Set procedure to other (takeoff)
        other = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 100)
        self.data.at[other, self.procedure_column] = 'other'

        # Set procedure to NADP1 (takeoff)
        nadp1 = (self.data[self.class_column] >= 500) & (self.data[self.class_column] < 600)
        self.data.at[nadp1, self.procedure_column] = 'NADP1'

        # Set procedure to NADP2 (takeoff)
        nadp2 = (self.data[self.class_column] >= 600) & (self.data[self.class_column] < 900)
        self.data.at[nadp2, self.procedure_column] = 'NADP2'

        # Set procedure to normal (landing)
        normal = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 1100)
        self.data.at[normal, self.procedure_column] = 'normal'

        # Set procedure to reduced flaps (landing)
        reduced_flaps = (self.data[self.class_column] >= 1200) & (self.data[self.class_column] < 1300)
        self.data.at[reduced_flaps, self.procedure_column] = 'reduced_flaps'

        # Set weight to heavy (takeoff)
        heavy = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                (self.data[self.class_column].mod(10) == 0)
        self.data.at[heavy, self.weight_column] = 'heavy'

        # Set weight to medium (takeoff)
        medium = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                 (self.data[self.class_column].mod(10) >= 1) & (self.data[self.class_column].mod(10) <= 2)
        self.data.at[medium, self.weight_column] = 'medium'

        # Set weight to light (takeoff)
        light = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                (self.data[self.class_column].mod(10) == 3)
        self.data.at[light, self.weight_column] = 'light'

        # Set altitude to 2000ft (landing)
        ft2000 = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
                 (self.data[self.class_column].mod(10) == 0)
        self.data.at[ft2000, self.altitude_column] = '2000ft'

        # Set altitude to 3000ft (landing)
        ft3000 = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
                 (self.data[self.class_column].mod(10) == 1)
        self.data.at[ft3000, self.altitude_column] = '3000ft'

        # Set altitude to CDA (landing)
        cda = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
              (self.data[self.class_column].mod(10) == 9)
        self.data.at[cda, self.altitude_column] = 'CDA'

    def add_denem(self, date_column=None):
        """
        Add day (D), evening (E), night (N) and early morning (EM) classification to the traffic, based on the date.

        :param date_column: the column with the date to use.
        """

        # Select the date column to use
        date_column = date_column if date_column is not None else self.date_column

        # Add a phase column
        self.data[self.denem_column] = None

        # Check for early morning (EM)
        em = self.data[date_column].dt.hour == 6
        self.data.at[em, self.denem_column] = 'EM'

        # Check for day (D)
        d = np.logical_and(self.data[date_column].dt.hour >= 7, self.data[date_column].dt.hour < 19)
        self.data.at[d, self.denem_column] = 'D'

        # Check for evening (E)
        e = np.logical_and(self.data[date_column].dt.hour >= 19, self.data[date_column].dt.hour < 23)
        self.data.at[e, self.denem_column] = 'E'

        # Check for night (N)
        n = np.logical_or(self.data[date_column].dt.hour >= 23, self.data[date_column].dt.hour < 6)
        self.data.at[n, self.denem_column] = 'N'

    def add_den(self, date_column=None):
        """
        Add day (D), evening (E), and night (N) classification to the traffic, based on the date.

        :param date_column: the column with the date to use.
        """

        # Select the date column to use
        date_column = date_column if date_column is not None else self.date_column

        # Add a phase column
        self.data[self.den_column] = None

        # Check for day (D)
        d = np.logical_and(self.data[date_column].dt.hour >= 7, self.data[date_column].dt.hour < 19)
        self.data.at[d, self.den_column] = 'D'

        # Check for evening (E)
        e = np.logical_and(self.data[date_column].dt.hour >= 19, self.data[date_column].dt.hour < 23)
        self.data.at[e, self.den_column] = 'E'

        # Check for night (N)
        n = np.logical_or(self.data[date_column].dt.hour >= 23, self.data[date_column].dt.hour < 7)
        self.data.at[n, self.den_column] = 'N'

    def get_den_distribution(self, separate_by=None, id_column=None):
        """
        Get the day (D), evening (E), and night (N) distribution of the traffic.

        :param separate_by: the column to use for grouping the results, such as meteorological year or takeoff/landing.
        :param id_column: the column to use for counting the traffic rows.
        :return: the DEN distribution of the traffic with DEN on the index and separate_by values as columns if provided
        :rtype: pd.DataFrame
        """

        # Select the date column to use
        id_column = id_column if id_column is not None else self.id_column

        if separate_by is None:
            return self.data.groupby([self.den_column])[id_column].count()

        # Get the distribution
        distribution = self.data.groupby([separate_by, self.den_column])[id_column].count().reset_index(drop=False)

        # Reshape the distribution
        return distribution.set_index([self.den_column]).pivot(columns=separate_by).xs(id_column, axis=1, level=0)

    def get_denem_distribution(self, separate_by=None, id_column=None):
        """
        Get the day (D), evening (E), night (N), and early morning (EM) distribution of the traffic.

        :param separate_by: the column to use for grouping the results, such as meteorological year or takeoff/landing.
        :param id_column: the column to use for counting the traffic rows.
        :return: the DENEM distribution of the traffic with DENEM on the index and separate_by values as columns if
        provided
        :rtype: pd.DataFrame
        """

        # Select the date column to use
        id_column = id_column if id_column is not None else self.id_column

        if separate_by is None:
            return self.data.groupby([self.denem_column])[id_column].count()

        # Get the distribution
        distribution = self.data.groupby([separate_by, self.denem_column])[id_column].count().reset_index(drop=False)

        # Reshape the distribution
        return distribution.set_index([self.denem_column]).pivot(columns=separate_by).xs(id_column, axis=1, level=0)

    def get_season_distribution(self, id_column=None):
        """
        Get the traffic count for all season and takeoff/landing combinations.

        :param id_column: the column to use for counting the traffic rows.
        :return: the distribution of the traffic with season and landing/takeoff on the index and DENEM as columns
        :rtype: pd.DataFrame
        """

        # Select the date column to use
        id_column = id_column if id_column is not None else self.id_column

        # Get the distribution
        distribution = self.data.groupby(['season', 'LT', self.denem_column])[id_column].count().reset_index(drop=False)

        # Reshape the distribution
        distribution = distribution.set_index(['season', 'LT']).pivot(columns=self.denem_column).xs(id_column, axis=1,
                                                                                                    level=0)

        # Return the sorted distribution
        return distribution[['D', 'E', 'N', 'EM']]

    def get_procedure_distribution(self):
        """
        Get the traffic count for all altitude classes of arrivals and procedure classes of departures.

        :return: the distribution of the traffic for takeoff/landing with altitude on the index for arrivals and
        procedure on the index for departures.
        :rtype: pd.DataFrame, pd.DataFrame
        """

        # Get the arrivals
        arrivals = self.data[self.data['LT'] == 'L'].groupby(self.altitude_column)[self.id_column].count()

        # Get the departures
        departures = self.data[self.data['LT'] == 'T'].groupby(self.procedure_column)[self.id_column].count()

        return arrivals, departures
    
    def get_emissions(self,
                      ET,
                      TIM,
                      ACtypes,
                      ac_cat,
                      f_1ipv2=0,
                      f_3ipv4=0,
                      f_2ipv3=0,
                      f_APU400hz=0,
                      f_APU=0.75,
                      reference_traffic=None,
                      new_engine=None):

        trf = self.data

        # Alleen ICAO aircraft code en engine? Dus geen MTOW
        trf=self.data.loc[:,[self.aircraft_column,self.engine_column]]

        ###TODO: Risico dat NaN worden gemist door de groupby        
        trf=trf.groupby(trf.columns.tolist()).size().reset_index().rename(columns={0:'total'})
        trf=trf.rename(columns={"C_ac_type": "ac_type", "C_engine_type": "engine_type"})

        return emission_model(trf,
                              ET, 
                              TIM, 
                              ACtypes, 
                              ac_cat,
                              f_1ipv2,
                              f_3ipv4,
                              f_2ipv3,
                              f_APU400hz,
                              f_APU, 
                              reference_traffic,
                              new_engine)

    
    def get_runway_usage_statistics(self, period):
        """
        Aggregate the runway usage for the given period of the day and calculate the various statistics, including mean,
        median, minimum, maximum and standard deviation.

        :param str period: a regular expression for the period, e.g. 'D' or 'D|E|N'
        :rtype: pd.DataFrame
        """

        # # Define the supported types
        # supported_types = ['daisy.meteoyear']

        # # Check if a different type is provided
        # if self.type not in supported_types:
        #     # List the supported types as a string
        #     supported_types_string = ', '.join(supported_types)

        #     # Include 'or' after the last comma
        #     supported_types_string = ', or '.join(supported_types_string.rsplit(', ', 1))

        #     raise TypeError('This method is only supported for traffic aggregates of type {}, but {} is given'.format(
        #         supported_types_string, self.type))

        # Match the period
        data = self.data[self.data['d_den'].str.match(period)]

        # Calculate the total runway usage per operation per year
        data = data.groupby(['d_lt', 'd_runway', 'd_myear'])['total'].sum().reset_index()

        # Describe the various yearly scenarios per runway per type of operation
        return data.groupby(['d_lt', 'd_runway'])['total'].describe()


class TrafficAggregate(object):
    def __init__(self, data, aggregate_type=None,class_column='d_proc',procedure_column='procedure', altitude_column='altitude',
                 weight_column='weight',sector_column='sectoriaf'):
        
        self.data = data
        self.type = aggregate_type
        self.class_column = class_column
        self.procedure_column = procedure_column
        self.altitude_column = altitude_column
        self.weight_column = weight_column
        self.sector_column = sector_column
        
    def add_procedure(self):
        """
        Add the procedure to the traffic, based on the four-digit traffic class.
        """

        # Add a procedure, altitude and weight column
        self.data[self.procedure_column] = None
        self.data[self.altitude_column] = None
        self.data[self.weight_column] = None

        # Set procedure to other (takeoff)
        other = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 100)
        self.data.at[other, self.procedure_column] = 'NADP1'

        # Set procedure to NADP1 (takeoff)
        nadp1 = (self.data[self.class_column] >= 500) & (self.data[self.class_column] < 600)
        self.data.at[nadp1, self.procedure_column] = 'NADP1'

        # Set procedure to NADP2 (takeoff)
        nadp2 = (self.data[self.class_column] >= 600) & (self.data[self.class_column] < 900)
        self.data.at[nadp2, self.procedure_column] = 'NADP2'

        # Set procedure to normal (landing)
        normal = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 1100)
        self.data.at[normal, self.procedure_column] = 'normal'

        # Set procedure to reduced flaps (landing)
        reduced_flaps = (self.data[self.class_column] >= 1200) & (self.data[self.class_column] < 1300)
        self.data.at[reduced_flaps, self.procedure_column] = 'reduced_flaps'

        # Set weight to heavy (takeoff)
        heavy = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                (self.data[self.class_column].mod(10) == 0)
        self.data.at[heavy, self.weight_column] = 'heavy'

        # Set weight to medium (takeoff)
        medium = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                 (self.data[self.class_column].mod(10) >= 1) & (self.data[self.class_column].mod(10) <= 2)
        self.data.at[medium, self.weight_column] = 'medium'

        # Set weight to light (takeoff)
        light = (self.data[self.class_column] >= 0) & (self.data[self.class_column] < 1000) & \
                (self.data[self.class_column].mod(10) == 3)
        self.data.at[light, self.weight_column] = 'light'

        # Set altitude to 2000ft (landing)
        ft2000 = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
                 (self.data[self.class_column].mod(10) == 0)
        self.data.at[ft2000, self.altitude_column] = '2000ft'

        # Set altitude to 3000ft (landing)
        ft3000 = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
                 (self.data[self.class_column].mod(10) == 1)
        self.data.at[ft3000, self.altitude_column] = '3000ft'

        # Set altitude to CDA (landing)
        cda = (self.data[self.class_column] >= 1000) & (self.data[self.class_column] < 2000) & \
              (self.data[self.class_column].mod(10) == 9)
        self.data.at[cda, self.altitude_column] = 'CDA'
        
        
    def add_sector(self,routesector):
        
        self.data[self.sector_column] = None
        # MERGE
        self.data = self.data.merge(routesector, 
                      left_on='d_route',
                      right_on='route',
                      how='left')

        # rename 
        self.data[self.sector_column] = self.data['sector']
        self.data.drop(['sector','route'], axis=1)
        
        # check for empty sectors
        
        
    def get_runway_usage(self, period):
        """
        Aggregate the runway usage for the given period of the day.

        :param str period: a regular expression for the period, e.g. 'D' or 'D|E|N'
        :rtype: pd.DataFrame
        """

        # Define the supported types
        supported_types = ['daisy.meteoyear', 'daisy.mean']

        # Check if a different type is provided
        if self.type not in supported_types:
            # List the supported types as a string
            supported_types_string = ', '.join(supported_types)

            # Include 'or' after the last comma
            supported_types_string = ', or '.join(supported_types_string.rsplit(', ', 1))

            raise TypeError('This method is only supported for traffic aggregates of type {}, but {} is given'.format(
                supported_types_string, self.type))

        # Match the period
        data = self.data[self.data['d_den'].str.match(period)]

        # Sum the number of flights for each LT-runway combination
        data = data.groupby(['d_lt', 'd_runway'])['total'].sum().reset_index()

        # Return the pivoted dataframe
        return data.pivot('d_lt', 'd_runway', 'total')

    def get_runway_usage_statistics(self, period):
        """
        Aggregate the runway usage for the given period of the day and calculate the various statistics, including mean,
        median, minimum, maximum and standard deviation.

        :param str period: a regular expression for the period, e.g. 'D' or 'D|E|N'
        :rtype: pd.DataFrame
        """

        # Define the supported types
        supported_types = ['daisy.meteoyear']

        # Check if a different type is provided
        if self.type not in supported_types:
            # List the supported types as a string
            supported_types_string = ', '.join(supported_types)

            # Include 'or' after the last comma
            supported_types_string = ', or '.join(supported_types_string.rsplit(', ', 1))

            raise TypeError('This method is only supported for traffic aggregates of type {}, but {} is given'.format(
                supported_types_string, self.type))

        # Match the period
        data = self.data[self.data['d_den'].str.match(period)]

        # Calculate the total runway usage per operation per year
        data = data.groupby(['d_lt', 'd_runway', 'd_myear'])['total'].sum().reset_index()

        # Describe the various yearly scenarios per runway per type of operation
        return data.groupby(['d_lt', 'd_runway'])['total'].describe()

    def get_den_distribution(self, separate_by=None):

        if separate_by is None:
            return self.data.groupby(['d_den'])['total'].sum()

        # Get the distribution
        distribution = self.data.groupby([separate_by, 'd_den'])['total'].sum().reset_index(drop=False)

        # Reshape the distribution
        distribution = distribution.set_index(['d_den']).pivot(columns=separate_by).xs('total', axis=1, level=0)

        return distribution
    
    def get_denem_distribution(self, separate_by=None):
  
        # redefine d_den column based on scheduled times
        try:
            t = pd.to_datetime(self.data['d_schedule'],format='%H:%M')
            self.data.loc[t.dt.hour == 6,'d_den'] = 'EM'
        except:
            self.data.loc[self.data['d_schedule'] == 6,'d_den'] = 'EM'
        
        if separate_by is None:
            return self.data.groupby(['d_den'])['total'].sum()

        # Get the distribution
        distribution = self.data.groupby([separate_by, 'd_den'])['total'].sum().reset_index(drop=False)

        # Reshape the distribution
        distribution = distribution.set_index(['d_den']).pivot(columns=separate_by).xs('total', axis=1, level=0)

        return distribution
    
    def get_procedure_distribution(self):
        """
        Get the traffic count for all altitude classes of arrivals and procedure classes of departures.

        :return: the distribution of the traffic for takeoff/landing with altitude on the index for arrivals and
        procedure on the index for departures.
        :rtype: pd.DataFrame, pd.DataFrame
        """

        # Get the arrivals
        arrivals = self.data[self.data['d_lt'] == 'L'].groupby(self.altitude_column)['total'].sum()

        # Get the departures
        departures = self.data[self.data['d_lt'] == 'T'].groupby(self.procedure_column)['total'].sum()
        
        # Get the reduced flaps
        reduced_flaps = self.data[self.data['d_lt'] == 'L'].groupby(self.procedure_column)['total'].sum()

        return arrivals, departures, reduced_flaps
    
    
    def get_sector_distribution(self):
        """
        Get the traffic count for all altitude classes of arrivals and procedure classes of departures.

        :return: the distribution of the traffic for takeoff/landing with altitude on the index for arrivals and
        procedure on the index for departures.
        :rtype: pd.DataFrame, pd.DataFrame
        """

        # Get the arrivals
        sector = self.data.groupby(self.sector_column)['total'].sum()

        return sector

    def get_n_runway_preference_usage(self, rc_preferences):

        # Get the runway combination usage in the specified period
        rc_usage = self.data[(self.data['d_den'] == 'N') & (self.data['d_schedule'] != 6)]

        # get the runway combination preference for the specified period
        rc_preference = rc_preferences[rc_preferences['period'] == 'N']

        return self.get_runway_preference_usage(rc_usage, rc_preference)

    def get_deem_runway_preference_usage(self, rc_preferences):

        # Get the runway combination usage in the specified period
        rc_usage = self.data[(self.data['d_den'] != 'N') | (self.data['d_schedule'] == 6)]

        # get the runway combination preference for the specified period
        rc_preference = rc_preferences[rc_preferences['period'] != 'N']

        return self.get_runway_preference_usage(rc_usage, rc_preference)

    @staticmethod
    def get_runway_preference_usage(rc_usage, rc_preference):

        # Add the preferences to the runway combinations
        rc_usage = pd.merge(rc_usage, rc_preference, left_on='d_combination', right_on='combination', how='left')

        # Fill the unknowns with -
        rc_usage = rc_usage.fillna('-')

        # Calculate the usage per preference
        rc_preference_usage = rc_usage.groupby(['preference'])['total'].sum()

        # Replace '-' with 'other'
        rc_preference_usage.index = rc_preference_usage.index.str.replace('-', 'other')

        # Add total
        rc_preference_usage = rc_preference_usage.append(pd.Series(rc_preference_usage.sum(), index=['total']))

        # Add subtotal (preferences only)
        is_preferred = rc_preference_usage.index.str.isnumeric()
        rc_preference_usage = rc_preference_usage.append(
            pd.Series(rc_preference_usage[is_preferred].sum(), index=['subtotal']))

        # Change the order of the index
        is_preferred = rc_preference_usage.index.str.isnumeric()
        new_index_order = rc_preference_usage.index[is_preferred].tolist() + ['subtotal', 'other', 'total']
        rc_preference_usage = rc_preference_usage.reindex(index=new_index_order)

        # Calculate the relative preference usage
        rc_preference_usage_relative = (rc_preference_usage / rc_preference_usage['total'] * 100)

        # Get the preference
        return pd.concat([rc_preference_usage.rename('usage'), rc_preference_usage_relative.rename('relative usage')],
                         axis=1)

    def get_bracket(self, percentile=None):
        """
        Aggregate the data for each twenty minute bracket.

        :param float percentile: value between 0 <= percentile <= 1, the percentile to compute. See also
        pd.DataFrame.quantile
        :rtype: Bracket
        """

        # Define the supported types
        supported_types = ['daisy.phase', 'daisy.weekday', 'taf.sir']

        # Check if a different type is provided
        if self.type not in supported_types:
            # List the supported types as a string
            supported_types_string = ', '.join(supported_types)

            # Include 'or' after the last comma
            supported_types_string = ', or '.join(supported_types_string.rsplit(', ', 1))

            raise TypeError('This method is only supported for traffic aggregates of type {}, but {} is given'.format(
                supported_types_string, self.type))

        # Convert the d_schedule column to a timedelta
        dt = pd.to_datetime(self.data['d_schedule'], format="%H:%M") - pd.to_datetime("00:00", format="%H:%M")

        # Create a numeric value for each 20 minute time range
        brackets = np.mod(np.floor(dt / pd.to_timedelta(20, unit='m')).astype(int), 72)

        # Combine the brackets with the original data
        data = pd.concat([self.data, brackets.rename('bracket')], axis=1)

        # Add a total if none is present in the data
        if 'total' not in data:
            data['total'] = 1

        if 'd_date' not in data:
            # Sum the number of takeoffs/landings each bracket
            bracket_data = data.groupby(['d_lt', 'bracket'])['total'].sum()

        else:
            # Sum the number of takeoffs/landings each bracket
            bracket_by_date = data.groupby(['d_lt', 'bracket', 'd_date'])['total'].sum()

            # Use the mean by default, or use the percentile if specified
            if percentile is None:
                bracket_data = bracket_by_date.groupby(['d_lt', 'bracket']).mean()
            else:
                bracket_data = bracket_by_date.groupby(['d_lt', 'bracket']).quantile(percentile)

        # Return a bracket with reshaped data
        return Bracket(bracket_data.reset_index().pivot('d_lt', 'bracket', 'total'))
 
    def get_emissions(self,
                      ET,
                      TIM,
                      ACtypes,
                      ac_cat,
                      f_1ipv2=0,
                      f_3ipv4=0,
                      f_2ipv3=0,
                      f_APU400hz=0,
                      f_APU=0.75,
                      reference_traffic=None,
                      new_engine=None):

        # Define the supported types
        supported_types = ['daisy.mean', 'casper']

        # Check if a different type is provided
        if self.type not in supported_types:
            # List the supported types as a string
            supported_types_string = ', '.join(supported_types)

            # Include 'or' after the last comma
            supported_types_string = ', or '.join(supported_types_string.rsplit(', ', 1))

            raise TypeError('This method is only supported for traffic aggregates of type {}, but {} is given'.format(
                supported_types_string, self.type))
    
        trf = self.data

        if self.type == 'casper':
            ###TODO: weight???
            trf=self.data.loc[:,[self.aircraft_column,self.engine_column]]
            
            trf=trf.groupby(trf.columns.tolist()).size().reset_index().rename(columns={0:'total'})
            trf=trf.rename(columns={"C_ac_type": "ac_type", "C_engine_type": "engine_type"})
        elif self.type == 'daisy.mean':           
            # To keep NaN: convert it to string
            ###TODO: Ook voor engine en MTOW???
            trf['d_type'] = trf['d_type'].astype(str)
        
            # Aggregate traffic: keep aircraft and engine type
            ###TODO d_motor of MTT_engine_type en wat als ze beide voorkomen?
            columns = [c for c in trf.columns if c in ['d_type', 'd_motor', 'MTT_engine_type', 'd_mtow']]
            trf = trf.groupby(columns)['total'].sum().reset_index()

            trf=trf.rename(columns={"d_type": "ac_type", "d_motor": "engine_type", 'MTT_engine_type': "engine_type"})

        return emission_model(trf,
                              ET, 
                              TIM, 
                              ACtypes, 
                              ac_cat,
                              f_1ipv2,
                              f_3ipv4,
                              f_2ipv3,
                              f_APU400hz,
                              f_APU, 
                              reference_traffic,
                              new_engine)
    
    


    def get_HG(self,HGdbase,ac_cat=None):
        
        
        try: 
            # merge with ac_cat table
            t    = self.data.merge(ac_cat,left_on='d_type',
                                                 right_on='icao_aircraft',
                                                 how='left')
            # drop columns
            t    = self.loc[:,['d_proc','d_type','d_schedule','d_myear','total']]
        except: 

            t    = self.data
            
        #% add HGdbase
        t  = t.merge(HGdbase, 
                        left_on=['d_proc','d_ac_cat'], 
                        right_on=['profileType','aircraftType'],
                        how='left')
        
        # warning for missing clusters
        # check for nans
        if t['aircraftType'].isnull().values.any():
            # unique missing clusters
            clusters = t.loc[t['aircraftType'].isnull(),['d_ac_cat','d_proc']]
            
            # compute correctie factor:
            totaal = t['total'].sum()
            print(totaal)
            totaal_cluster = t.loc[t['aircraftType'].isnull(),'total'].sum()
            print(totaal_cluster)
            cf = totaal/(totaal-totaal_cluster)
            print(cf)
            # get rid of missing clusters
            clusters_unique = clusters.drop_duplicates()
            print('WARNING: '+str(totaal_cluster) +' flights with missing clusters in HG database:')
            print(clusters_unique.values)

        #% straffactoren
        ids_E = (t['d_schedule']>18) & (t['d_schedule']<23) 
        ids_N = (t['d_schedule']>22) | (t['d_schedule']<7) 
        
        t.loc[ids_E,'dBlin'] = t.loc[ids_E,'dBlin']*np.sqrt(10)
        t.loc[ids_N,'dBlin'] = t.loc[ids_N,'dBlin']*10
        
        # now multiply with the total number of movements.
        t['dBlin_totaal']= t['dBlin']*t['total']
        
        # nieuwe meteomarge op de HG
        t = t.groupby(['d_myear']).sum()
        
        # get exceptional years
        exceptional_years =[1981, 1984, 1993, 1994, 1996, 2000, 2002, 2010] 
        years = np.setdiff1d(np.arange(1971, 2011), exceptional_years)
        t =  np.amax(t.loc[years,:])

        # Computing HG per meteoyear
        HGlin=t['dBlin_totaal']*cf
        HG=10*math.log10(HGlin)
        
        return HG
    
class Bracket(object):

    def __init__(self, data):
        self.data = data

    @classmethod
    def read_taf_bracket_excel_file(cls, path, **kwargs):
        """
        A method to read TAF bracket files.

        :param str path: path to the file.
        :return: Bracket.
        :rtype: Bracket
        """

        # Get the data from the file
        data = pd.read_excel(path, **kwargs)

        # Use a zero based index for the bracket numbers
        data['bracket'] = data['bracket'] - data['bracket'].min()

        # Reshape the bracket
        data = data.set_index('bracket').T

        return cls(data)

    @classmethod
    def from_periods_and_capacity(cls, period, capacity):
        """
        A method to read Daisy period files.

        :param str path: path to the file.
        :return: Bracket.
        :rtype: Bracket
        """

        # Get the dominant period
        period['period'] = period['period'].str.split(',', 1).str[0]

        # Merge the capacity with the periods and rename the
        period = period.merge(capacity, on='period', how='left').rename(columns={'Lcap': 'L', 'Tcap': 'T'})

        # Reshape the bracket
        period = period[['L', 'T', 'period']].T

        return cls(period)


def start_summer_season(year):
    """
    Determine the start of the summer season, which is the last Sunday of the month March.

    :param int year: the calendar year of the season
    :return the start date of the summer season
    :rtype pd.Timestamp
    """

    # Get the last day of March
    last_day = pd.Timestamp(year=year, month=3, day=31)

    # Return the last Sunday of March
    return last_day - pd.Timedelta((last_day.weekday() + 1) % 7, unit='day')


def start_winter_season(year):
    """
    Determine the start of the summer season, which is the last Sunday of the month October.

    :param int year: the calendar year of the season
    :return the start date of the summer season
    :rtype datetime.date
    """

    # Get the last day of October
    last_day = pd.Timestamp(year=year, month=10, day=31)

    # Return the last Sunday of March
    return last_day - pd.Timedelta((last_day.weekday() + 1) % 7, unit='day')


def get_year_of_use(year):
    """
    Determine start- and end date, number of days in the year of use.
    """

    # Create a dictionary for the information
    year_info = {
        'year': year,
        'start_summer': start_summer_season(year),
        'end_summer': start_winter_season(year) + datetime.timedelta(-1),
        'start_winter': start_winter_season(year - 1),
        'end_winter': start_summer_season(year)
    }

    # Number of days, weeks
    year_info['winter_days'] = (year_info['end_winter'] - year_info['start_winter']).days + 1
    year_info['summer_days'] = (year_info['end_summer'] - year_info['end_winter']).days
    year_info['winter_weeks'] = year_info['winter_days'] / 7
    year_info['summer_weeks'] = year_info['summer_days'] / 7

    return year_info
