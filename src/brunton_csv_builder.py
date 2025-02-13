import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
import pyvista as pv
import gemgis as gg

class observations(object):
    
    ''' This class is the organizational component for stochasitic modeling
    
    
    Parameters:
       data_path (str) : path to working directory (include a trailing '/') e.g "/Users/name/working_directory/"
       init_interfaces (str) : initial interfaces CSV path
       '''
    
    def __init__(self, data_path='./', output_path=None, init_interfaces=None, init_orients=None, output_prefix='No_entry'):
        

        '''
        The init will create a dataframe to populate with the adding functions included in this class.

        You can import a CSV initially from file by including the path as an argument for "init_interfaces" or "init_orients"

        setting the data_path and output_path act as a prefix for each input/output function.

        ''' 


        # output parameters
        self.output_path = output_path  
        self.output_prefix = output_prefix
        
        # input parameters
        self.init_interfaces = init_interfaces
        self.init_orients = init_orients
        self.data_path = data_path

        #combining file path and input to use internally in the object
    

        # Load the inital CSV file if they exist
        try:
            if init_interfaces != None:
                self.interfaces_path = self.data_path+self.init_interfaces
                self.interfaces =  pd.read_csv(self.interfaces_path)
            else:
                self.interfaces = pd.DataFrame(None)
        except Exception as e:
            raise ValueError(f"Error reading the interfaces CSV file: {e}")(self.interfaces_path)
        
        try:
            if init_orients != None:
                self.orients_path = self.data_path+self.init_orients
                self.orients =  pd.read_csv(self.orients_path)
            else:
                self.orients = pd.DataFrame(None)
        except Exception as e:
            raise ValueError(f"Error reading the interfaces CSV file: {e}")(self.orients_path)

        self.icols = ['X', 'Y', 'Z','col','row', 'depth', 'depth1', 'depth2']

        
        self.ocols = ["input_ID", 'input_type', 'correlation', 'doi', 'source',
                        "X", "Y", "Z", 'formation', 'azimuth', 'dip', 'polarity', 
                        'X_variance', 'Y_variance', 'Z_variance', 'azimuth_variance', 'dip_variance', 
                        'min_depth_confidence','max_depth_confidence']
         

        self.input_id_counter = 1 

    def show_current_stats(self):

        '''
        Prints output of the current stats
        '''

        print("Oberservations_collected")
        print(f'Output path: {self.output_path}')
        print(f'Output Prefix: {self.output_prefix}')

        print(f"CSV import columns: {self.icols}")
        print("Input interfaces head:")
        print(self.interfaces.head(5))

        
        print(f"CSV import columns: {self.ocols}")
        print("Input orients head:")
        print(self.orients.head(5))

    def update_csv_icols(self, cols):
        
        self.icols = cols
        print(f"CSV import columns: {cols}")
    
    def update_csv_ocols(self, cols):
        
        self.ocols = cols
        print(f"CSV import columns: {cols}")




    # __________ pole to vector and reverse conversion functions _____________
    def dip_azimuth_to_pole_vector(dip, azimuth):
        """
        Converts dip and azimuth to a pole (normal) vector.
        
        Parameters:
            dip (float): Dip angle in degrees (0° to 90°).
            azimuth (float): Azimuth angle in degrees (0° to 360°).
        
        Returns:
            np.ndarray: Pole vector as [x, y, z].
        """
        # Convert angles from degrees to radians
        dip_rad = np.radians(dip)
        azimuth_rad = np.radians(azimuth)
        
        # Compute the components of the pole vector
        x = np.sin(dip_rad) * np.sin(azimuth_rad)
        y = np.sin(dip_rad) * np.cos(azimuth_rad)
        z = np.cos(dip_rad)
        
        return np.array([x, y, z])


    def update_dataframe_with_dip_azimuth(self, df):
        """
        Update a pandas DataFrame by applying the pole_vector_to_dip_azimuth function to three-component vectors.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            pole_vector_to_dip_azimuth (function): Function that takes [nx, ny, nz] and returns (azimuth, dip).
            nx_col (str): Column name for the x-component.
            ny_col (str): Column name for the y-component.
            nz_col (str): Column name for the z-component.
            azimuth_col (str): Column name to store the azimuth.
            dip_col (str): Column name to store the dip.

        Returns:
            pd.DataFrame: Updated DataFrame with azimuth and dip columns.
        """
        def pole_vector_to_dip_azimuth(pole_vector):
            """
            Converts a pole (X,Y,Z) normal vector to dip and azimuth.
            
            Parameters:
                pole_vector (np.ndarray): Pole vector as [x, y, z].
            
            Returns:
                tuple: (dip in degrees, azimuth in degrees)
            """
            # Normalize the vector to ensure it's a unit vector
            pole_vector = pole_vector / np.linalg.norm(pole_vector)
            x, y, z = pole_vector
            
            # Compute dip angle (inverse cosine of the z component)
            dip = np.degrees(np.arccos(z))
            
            # Compute azimuth (angle in the x-y plane)
            azimuth = np.degrees(np.arctan2(x, y))
            
            # Adjust azimuth to lie between 0° and 360°
            azimuth = azimuth % 360
            
            return azimuth, dip
        
        def compute_azimuth_dip(row):
            # Extract the vector components from the row
            nx, ny, nz = row['nx'], row['ny'], row['nz']
            # Apply the input function
            azimuth_col, dip_col = pole_vector_to_dip_azimuth([nx, ny, nz])
            return pd.Series({azimuth_col:'azimuth', dip_col: 'dip'})

        # Apply the function row-wise and update the DataFrame
        df[['azimuth', 'dip']] = df.apply(compute_azimuth_dip, axis=1)
        #print(df.head(10))
        return df

    # ____________ THESE ARE THE ADDING FUNCTIONS _______________

    def add_csv_to_interfaces(self, file_list=[], **kwargs):
        '''
        input [list]: this is a list of files that you want to add
        '''
        _sample_size = kwargs.get('sample_size')
        if _sample_size==None:
            _sample_size=1
    
        for file in file_list:
            print(f"adding: {self.data_path+file}")
            csv_to_add = pd.read_csv(self.data_path+file, names=self.icols)
            #csv_to_add = pd.read_csv(self.data_path+file, sep=' ', names=self.icols, skiprows=np.arange(20))
            self.interfaces = pd.concat((self.interfaces,csv_to_add.sample(frac=_sample_size)))
            

        print(f"Completed adding {len(file_list)} CSV's")

    def add_csv_to_orients(self, file_list=[], **kwargs):
        '''
        input [list]: this is a list of files that you want to add
        '''
        _sample_size = kwargs.get('sample_size')
        if _sample_size==None:
            _sample_size=1
            
    
        for file in file_list:
            csv_to_add = pd.read_csv(self.data_path+file, names=self.ocols)
            self.orients = pd.concat((self.orients,csv_to_add.sample(frac=_sample_size)))
            

        print(f"Completed adding {len(file_list)} CSV's")


    def add_shapefile_to_interfaces(self, file_list, zdepth=None, formation='NOT_ENTERED', azimuth=None, dip=None, **kwargs):



        #Keyword Args assingment

        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')

        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')


        azimuth_field = kwargs.get('azimuth_field')
        dip_field = kwargs.get('dip_field')
        formation_field = kwargs.get('formation_field')
        z_field = kwargs.get('z_field') 
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar') 
        dip_variance = kwargs.get('dipvar') 
        azimuth_variance = kwargs.get('azivar')
        _sample_size = kwargs.get('sample_size')
        if _sample_size==None:
            _sample_size=1
        

        for file in file_list:
            shp_to_add = gpd.read_file(self.data_path+file)
            shape_template = pd.DataFrame(None)
            
            #Metadata logic tree
            if input_id != None:
                shape_template['input_ID'] = input_id
            else:
                shape_template['input_ID'] = self.input_id_counter
                self.input_id_counter = self.input_id_counter+1

            if input_type != None:
                shape_template['input_type'] = input_type
            else:
                shape_template['input_type'] = 'Unknown'
            
            if self_correlation != None:
                shape_template['correlation'] = self_correlation
            else:
                shape_template['correlation'] = 0

            if doi != None:
                shape_template['doi'] = doi
            else:
                shape_template['doi'] = 'no doi'

            if source != None:
                shape_template['source'] = source
            else:
                shape_template['source'] = 'no souce'

            #point data logic tree
            shape_template['X'] = shp_to_add.geometry.x
            shape_template['Y'] = shp_to_add.geometry.y
            if zdepth==None:
                shape_template['Z'] = shp_to_add.geometry.z
            elif z_field==None:    
                shape_template['Z'] = zdepth
            else:
                shape_template['Z'] = shp_to_add[z_field]
            
            if formation_field != None:
                shape_template['formation'] = shp_to_add[formation_field]
            else:
                shape_template['formation'] = formation
            if azimuth_field != None:
                shape_template['azimuth'] = shp_to_add[azimuth_field]
            else:
                shape_template['azimuth'] = azimuth
            if dip_field != None:
                shape_template['dip'] = shp_to_add[dip_field]
            else:
                shape_template['dip'] = dip

            if X_variance != None:
                shape_template['X_variance'] = X_variance
            else:
                shape_template['X_variance'] = 1

            if Y_variance != None:
                shape_template['Y_variance'] = Y_variance
            else:
                shape_template['Y_variance'] = 1

            if Z_variance != None:
                shape_template['Z_variance'] = Z_variance
            else:
                shape_template['Z_variance'] = 1
            
            if azimuth_variance != None:
                shape_template['azimuth_variance'] = azimuth_variance
            else:
                shape_template['azimuth_variance'] = 1
            
            if dip_variance != None:
                shape_template['dip_variance'] = dip_variance
            else:
                shape_template['dip_variance'] = 1
            if min_conf != None:
                shape_template['min_depth_confidence'] = min_conf
            else:
                shape_template['min_depth_confidence'] = None
            
            if max_conf != None:
                shape_template['max_depth_confidence'] = max_conf
            else:
                shape_template['max_depth_confidence'] = None 

            shape_template['polarity'] = 1
            #shape_template['azimuth'] = azimuth
            #shape_template['dip'] = dip
            
            self.interfaces = pd.concat((self.interfaces,shape_template.sample(frac=_sample_size)))
        
        print(f"Completed adding {len(file_list)} shapefile(s) to interfaces")
    

    def add_shapefile_to_orients(self, file_list, zdepth=None, formation='NOT_ENTERED', azimuth_field=None, set_azimuth=None, dip=None, **kwargs):

        #Keyword Args assingment
        
        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')

        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')


        #Keyword Args assingment
        azimuth_field = kwargs.get('azimuth_field')
        dip_field = kwargs.get('dip_field')
        formation_field = kwargs.get('formation_field')
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar') 
        dip_variance = kwargs.get('dipvar') 
        azimuth_variance = kwargs.get('azivar')
        _sample_size = kwargs.get('sample_size')
        if _sample_size==None:
            _sample_size=1
        

        for file in file_list:
            shp_to_add = gpd.read_file(self.data_path+file)
            shape_template = pd.DataFrame(None)
            #Metadata logic tree
            if input_id != None:
                shape_template['input_ID'] = input_id
            else:
                shape_template['input_ID'] = self.input_id_counter
                self.input_id_counter = self.input_id_counter+1

            if input_type != None:
                shape_template['input_type'] = input_type
            else:
                shape_template['input_type'] = 'Unknown'
            
            if self_correlation != None:
                shape_template['correlation'] = self_correlation
            else:
                shape_template['correlation'] = 0

            if doi != None:
                shape_template['doi'] = doi
            else:
                shape_template['doi'] = 'no doi'

            if source != None:
                shape_template['source'] = source
            else:
                shape_template['source'] = 'no souce'

            #point data logic tree
            shape_template['X'] = shp_to_add.geometry.x
            shape_template['Y'] = shp_to_add.geometry.y
            if zdepth==None:
                shape_template['Z'] = shp_to_add.geometry.z
            else:    
                shape_template['Z'] = zdepth
            if formation_field != None:
                shape_template['formation'] = shp_to_add[formation_field]
            else:
                shape_template['formation'] = formation
            if azimuth_field != None:
                shape_template['azimuth'] = shp_to_add[azimuth_field]
            else:
                shape_template['azimuth'] = set_azimuth
            if dip_field != None:
                shape_template['dip'] = shp_to_add[dip_field]
            else:
                shape_template['dip'] = dip

            if X_variance != None:
                shape_template['X_variance'] = X_variance
            else:
                shape_template['X_variance'] = 1

            if Y_variance != None:
                shape_template['Y_variance'] = Y_variance
            else:
                shape_template['Y_variance'] = 1

            if Z_variance != None:
                shape_template['Z_variance'] = Z_variance
            else:
                shape_template['Z_variance'] = 1
            
            if azimuth_variance != None:
                shape_template['azimuth_variance'] = azimuth_variance
            else:
                shape_template['azimuth_variance'] = 1
            
            if dip_variance != None:
                shape_template['dip_variance'] = dip_variance
            else:
                shape_template['dip_variance'] = 1
            
            if min_conf != None:
                shape_template['min_depth_confidence'] = min_conf
            else:
                shape_template['min_depth_confidence'] = None
            
            if max_conf != None:
                shape_template['max_depth_confidence'] = max_conf
            else:
                shape_template['max_depth_confidence'] = None 


            shape_template['polarity'] = 1
            #shape_template['azimuth'] = azimuth
            #shape_template['dip'] = dip
            
            self.orients = pd.concat((self.orients,shape_template.sample(frac=_sample_size)))
        
        print(f"Completed adding {len(file_list)} shapefile(s) to orients")


    def sample_surface_from_points(self, points, method='grid', num_points=100, grid_spacing=10, flip_normal=False, auto_orient_normals=False, **kwargs):
        """
        Construct a surface from input points, compute normals, and sample points.

        Parameters:
        - points (np.ndarray): Array of shape (N, 3) containing the input points [x, y, z].
        - method (str): Sampling method, either "random" or "grid".
        - num_points (int): Number of random points to sample (used if method is "random").
        - grid_spacing (float): Spacing for regularized grid (used if method is "grid").

        Returns:
        - pd.DataFrame: DataFrame with columns ["x", "y", "z", "nx", "ny", "nz"].
        """

        #Keyword Args assingment


        # Create a PyVista point cloud
        point_cloud = pv.PolyData(points)

        # Create a surface from the point cloud using Delaunay 2D triangulation
        surface = point_cloud.delaunay_2d()

        # Compute normals for the surface
        surface = surface.compute_normals(auto_orient_normals=auto_orient_normals, point_normals=True, flip_normals=flip_normal)

        # Extract bounds of the surface
        bounds = surface.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        minx, maxx, miny, maxy, _, _ = bounds

        sampled_points = []

        if method == "random":
        # Generate random points within the bounds
            for _ in range(num_points):
                x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
                z = 0  # Assume z = 0 for the surface elevation (or adjust as needed)
                sampled_point_index = surface.find_closest_point((x, y, z))  # Corrected to tuple
                point = surface.points[sampled_point_index]
                normal = surface.point_normals[sampled_point_index]
                sampled_points.append((*point, *normal, 0, 0))

        elif method == "grid":
            # Generate a regular grid of points within the bounds
            x_coords = np.arange(minx, maxx, grid_spacing)
            y_coords = np.arange(miny, maxy, grid_spacing)
            for x in x_coords:
                for y in y_coords:
                    z = 0  # Assume z = 0 for the surface elevation (or adjust as needed)
                    sampled_point_index = surface.find_closest_point((x, y, z))  # Corrected to tuple
                    point = surface.points[sampled_point_index]
                    normal = surface.point_normals[sampled_point_index]
                    sampled_points.append((*point, *normal, 0, 0))
        else:
            raise ValueError("Method must be either 'random' or 'grid'.")

        # Prepare the DataFrame
        df = pd.DataFrame(sampled_points, columns=["X", "Y", "Z", "nx", "ny", "nz", 'azimuth', 'dip'])
        #df = self.update_dataframe_with_dip_azimuth(df)

        def pole_vector_to_dip_azimuth(pole_vector):
            """
            Converts a pole (X,Y,Z) normal vector to dip and azimuth.
            
            Parameters:
                pole_vector (np.ndarray): Pole vector as [x, y, z].
            
            Returns:
                tuple: (dip in degrees, azimuth in degrees)
            """
            
            # Normalize the vector to ensure it's a unit vector
            pole_vector = pole_vector / np.linalg.norm(pole_vector)
            x, y, z = pole_vector
            
            # Compute dip angle (inverse cosine of the z component)
            dip = np.degrees(np.arccos(z))
            
            # Compute azimuth (angle in the x-y plane)
            azimuth = np.degrees(np.arctan2(x, y))
            
            # Adjust azimuth to lie between 0° and 360°
            azimuth = azimuth % 360
            
            return azimuth, dip
        def compute_azimuth_dip(row):
            # Extract the vector components from the row
            nx, ny, nz = row['nx'], row['ny'], row['nz']
            # Apply the input function
            azimuth, dip = pole_vector_to_dip_azimuth([nx, ny, nz])
            return pd.Series({'azimuth': azimuth, 'dip': dip})

        # Apply the function row-wise and update the DataFrame
        df[['azimuth', 'dip']] = df.apply(compute_azimuth_dip, axis=1)
            #nx, ny, nz = row['nx'], row['ny'], row['nz']
                # Apply the input function
        #df['azimuth', 'dip'] = df['nx','ny','nz'].apply(pole_vector_to_dip_azimuth, axis=1)
        return df

    def add_surface_points_to_orients(self, surface, sample_method='grid', num_points=100, grid_spacing=10, polarity=1, auto_orient_normals=True, flip_normal=False, **kwargs):
        
        #Keyword Args assingment
        
        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')
        
        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')

        #Assigning KWargs
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar') 
        dip_variance = kwargs.get('dipvar') 
        azimuth_variance = kwargs.get('azivar')  
        
        formation = kwargs.get('formation')
        if formation == None:
            formation = 'NOT_ENTERED'
        frac = kwargs.get('frac')
        if frac ==None:
            frac=1
        if sample_method == 'grid':
            method='grid'
        elif sample_method == 'random':
            method='random'

        num_points=num_points
        grid_spacing=grid_spacing
        gdf = gpd.read_file(surface)
        surf_ = []
        for point in gdf.geometry:
            surf_.append((point.x,point.y,point.z))


        df = self.sample_surface_from_points(surf_, method, num_points, grid_spacing, auto_orient_normals=auto_orient_normals, flip_normal=flip_normal)
        df['formation'] = formation
        
        df['polarity'] = polarity

        #Metadata logic tree
        if input_id != None:
            df['input_ID'] = input_id
        else:
            df['input_ID'] = self.input_id_counter
            self.input_id_counter = self.input_id_counter+1

        if input_type != None:
            df['input_type'] = input_type
        else:
            df['input_type'] = 'Unknown'
            
        if self_correlation != None:
            df['correlation'] = self_correlation
        else:
            df['correlation'] = 0

        if doi != None:
            df['doi'] = doi
        else:
            df['doi'] = 'no doi'

        if source != None:
            df['source'] = source
        else:
            df['source'] = 'no source'

        #point data logic tree        
        
        if X_variance != None:
            df['X_variance'] = X_variance
        else:
            df['X_variance'] = 1

        if Y_variance != None:
            df['Y_variance'] = Y_variance
        else:
            df['Y_variance'] = 1

        if Z_variance != None:
            df['Z_variance'] = Z_variance
        else:
            df['Z_variance'] = 1
            
        if azimuth_variance != None:
            df['azimuth_variance'] = azimuth_variance
        else:
            df['azimuth_variance'] = 1
            
        if dip_variance != None:
            df['dip_variance'] = dip_variance
        else:
            df['dip_variance'] = 1
        
        if min_conf != None:
            df['min_depth_confidence'] = min_conf
        else:
            df['min_depth_confidence'] = None
            
        if max_conf != None:
            df['max_depth_confidence'] = max_conf
        else:
            df['max_depth_confidence'] = None


        self.orients = pd.concat((self.orients,df.sample(frac=frac)))



    def add_surface_points_to_interfaces(self, surface, sample_method='grid', num_points=100, grid_spacing=10, polarity=1, auto_orient_normals=True, flip_normal=False, **kwargs):
        

        
        #Keyword Args assingment
        
        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')

        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')
        
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar') 
        dip_variance = kwargs.get('dipvar') 
        azimuth_variance = kwargs.get('azivar')  
        
        formation = kwargs.get('formation')
        if formation == None:
            formation = 'NOT_ENTERED'
        frac = kwargs.get('frac')
        if frac ==None:
            frac=1
        if sample_method == 'grid':
            method='grid'
        elif sample_method == 'random':
            method='random'

        num_points=num_points
        grid_spacing=grid_spacing
        gdf = gpd.read_file(surface)
        surf_ = []
        for point in gdf.geometry:
            surf_.append((point.x,point.y,point.z))
            
        df = self.sample_surface_from_points(surf_, method, num_points, grid_spacing, auto_orient_normals=auto_orient_normals, flip_normal=flip_normal)
        df['formation'] = formation
        df['polarity'] = polarity

        #Metadata logic tree
        if input_id != None:
            df['input_ID'] = input_id
        else:
            df['input_ID'] = self.input_id_counter
            self.input_id_counter = self.input_id_counter+1

        if input_type != None:
            df['input_type'] = input_type
        else:
            df['input_type'] = 'Unknown'
            
        if self_correlation != None:
            df['correlation'] = self_correlation
        else:
            df['correlation'] = 0

        if doi != None:
            df['doi'] = doi
        else:
            df['doi'] = 'no doi'

        if source != None:
            df['source'] = source
        else:
            df['source'] = 'no source'

        #point data logic tree

        if X_variance != None:
            df['X_variance'] = X_variance
        else:
            df['X_variance'] = 1

        if Y_variance != None:
            df['Y_variance'] = Y_variance
        else:
            df['Y_variance'] = 1

        if Z_variance != None:
            df['Z_variance'] = Z_variance
        else:
            df['Z_variance'] = 1
            
        if azimuth_variance != None:
            df['azimuth_variance'] = azimuth_variance
        else:
            df['azimuth_variance'] = 1
            
        if dip_variance != None:
            df['dip_variance'] = dip_variance
        else:
            df['dip_variance'] = 1
        
        if min_conf != None:
            df['min_depth_confidence'] = min_conf
        else:
            df['min_depth_confidence'] = None
            
        if max_conf != None:
            df['max_depth_confidence'] = max_conf
        else:
            df['max_depth_confidence'] = None 

        self.interfaces = pd.concat((self.interfaces,df.sample(frac=frac)))

    def add_shapefile_linestrings_and_compute_azimuth_to_orients(self, filename, fraction=1, z=0, azimuth_reverse=True, **kwargs):
        #Keyword Args assingment
        
        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')

        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')

        dip = kwargs.get('dip')
        formation = kwargs.get('formation')
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar') 
        dip_variance = kwargs.get('dipvar') 
        azimuth_variance = kwargs.get('azivar')
        
        file = self.data_path+filename
        inputgdf = gpd.read_file(file)
        inputgdf['dZ'] = 0
        #print(inputgdf.head(3))
        shape_template = gg.vector.extract_orientations_from_map(gdf=inputgdf)
        shape_template['Z'] = z
        if formation != None:
            shape_template['formation'] = formation
        else:
            shape_template.rename(columns={'Name': 'formation'}, inplace=True)

        #Metadata logic tree
        if input_id != None:
            shape_template['input_ID'] = input_id
        else:
            shape_template['input_ID'] = self.input_id_counter
            self.input_id_counter = self.input_id_counter+1

        if input_type != None:
            shape_template['input_type'] = input_type
        else:
            shape_template['input_type'] = 'Unknown'
            
        if self_correlation != None:
            shape_template['correlation'] = self_correlation
        else:
            shape_template['correlation'] = 0

        if doi != None:
            shape_template['doi'] = doi
        else:
            shape_template['doi'] = 'no doi'

        if source != None:
            shape_template['source'] = source
        else:
            shape_template['source'] = 'no source'

            #point data variance logic tree
        if X_variance != None:
            shape_template['X_variance'] = X_variance
        else:
            shape_template['X_variance'] = 1

        if Y_variance != None:
            shape_template['Y_variance'] = Y_variance
        else:
            shape_template['Y_variance'] = 1

        if Z_variance != None:
            shape_template['Z_variance'] = Z_variance
        else:
            shape_template['Z_variance'] = 1
            
        if azimuth_variance != None:
            shape_template['azimuth_variance'] = azimuth_variance
        else:
            shape_template['azimuth_variance'] = 1
            
        if dip_variance != None:
            shape_template['dip_variance'] = dip_variance
        else:
            shape_template['dip_variance'] = 1

        if min_conf != None:
            shape_template['min_depth_confidence'] = min_conf
        else:
            shape_template['min_depth_confidence'] = None
            
        if max_conf != None:
            shape_template['max_depth_confidence'] = max_conf
        else:
            shape_template['max_depth_confidence'] = None 

        df = pd.DataFrame(shape_template, columns=["input_ID", 'input_type', 'correlation', 'doi', 'source',
                        "X", "Y", "Z", 'formation', 'azimuth', 'dip', 'polarity', 
                        'X_variance', 'Y_variance', 'Z_variance', 'azimuth_variance', 'dip_variance', 
                        'min_depth_confidence','max_depth_confidence'])
        if azimuth_reverse:
            df['azimuth'] = df['azimuth']+90 # to orient vector 90 deg from the map azimuth this should really only be applied to faults 
        else:
            df['azimuth'] = df['azimuth']-90 # to orient vector 90 deg from the map azimuth this should really only be applied to faults
        df['dip'] = dip 
        self.orients = pd.concat((self.orients,df.sample(frac=fraction)))


    def add_shapefile_linestrings_and_compute_azimuth_to_interfaces(self, filename, fraction=1, z=0, azimuth_reverse=True, **kwargs):
        #Keyword Args assingment
        
        # metadata
        input_id = kwargs.get('input_id')
        input_type = kwargs.get('input_type')
        self_correlation = kwargs.get('self_correlation')
        doi = kwargs.get('doi')
        source = kwargs.get('source')

        #confidence elevation in True vertical depth
        min_conf = kwargs.get('min_depth_confidence')
        max_conf = kwargs.get('max_depth_confidence')

        dip = kwargs.get('dip')
        formation = kwargs.get('formation')
        X_variance = kwargs.get('xvar')
        Y_variance = kwargs.get('yvar') 
        Z_variance = kwargs.get('zvar')
        dip_variance = kwargs.get('dipvar')
        azimuth_variance = kwargs.get('azivar')
        
        file = self.data_path+filename
        inputgdf = gpd.read_file(file)
        inputgdf['dZ'] = 0
        #print(inputgdf.head(3))
        shape_template = gg.vector.extract_orientations_from_map(gdf=inputgdf)
        shape_template['Z'] = z
        if formation != None:
            shape_template['formation'] = formation
        else:
            shape_template.rename(columns={'Name': 'formation'}, inplace=True)
        
        #Metadata logic tree
        if input_id != None:
            shape_template['input_ID'] = input_id
        else:
            shape_template['input_ID'] = self.input_id_counter
            self.input_id_counter = self.input_id_counter+1

        if input_type != None:
            shape_template['input_type'] = input_type
        else:
            shape_template['input_type'] = 'Unknown'
            
        if self_correlation != None:
            shape_template['correlation'] = self_correlation
        else:
            shape_template['correlation'] = 0

        if doi != None:
            shape_template['doi'] = doi
        else:
            shape_template['doi'] = 'no doi'

        if source != None:
            shape_template['source'] = source
        else:
            shape_template['source'] = 'no source'

            #point data variance logic tree
        if X_variance != None:
            shape_template['X_variance'] = X_variance
        else:
            shape_template['X_variance'] = 1

        if Y_variance != None:
            shape_template['Y_variance'] = Y_variance
        else:
            shape_template['Y_variance'] = 1

        if Z_variance != None:
            shape_template['Z_variance'] = Z_variance
        else:
            shape_template['Z_variance'] = 1
            
        if azimuth_variance != None:
            shape_template['azimuth_variance'] = azimuth_variance
        else:
            shape_template['azimuth_variance'] = 1
            
        if dip_variance != None:
            shape_template['dip_variance'] = dip_variance
        else:
            shape_template['dip_variance'] = 1

        if min_conf != None:
            shape_template['min_depth_confidence'] = min_conf
        else:
            shape_template['min_depth_confidence'] = None
            
        if max_conf != None:
            shape_template['max_depth_confidence'] = max_conf
        else:
            shape_template['max_depth_confidence'] = None     
        

        df = pd.DataFrame(shape_template, columns=["input_ID", 'input_type', 'correlation', 'doi', 'source',
                        "X", "Y", "Z", 'formation', 'azimuth', 'dip', 'polarity', 
                        'X_variance', 'Y_variance', 'Z_variance', 'azimuth_variance', 'dip_variance', 
                        'min_depth_confidence','max_depth_confidence'])
        
        if azimuth_reverse:
            df['azimuth'] = df['azimuth']+90 # to orient vector 90 deg from the map azimuth this should really only be applied to faults 
        else:
            df['azimuth'] = df['azimuth']-90 # to orient vector 90 deg from the map azimuth this should really only be applied to faults
        df['dip'] = dip 
        self.interfaces = pd.concat((self.interfaces,df.sample(frac=fraction)))

    # _____________ EXPORTING AND VISUALIZATION FUNCTIONS ________________


    def export_interfaces(self, filename="interfaces.csv"):
        export_path = self.output_path+self.output_prefix+filename
        self.interfaces.to_csv(export_path)
        print(f"completed exporting: {export_path}")

    def export_orientations(self, filename="orients.csv"):
        export_path = self.output_path+self.output_prefix+filename
        self.orients.to_csv(export_path)
        print(f"completed exporting: {export_path}")
    
    def display_3D(self):   #This is not working well yet
        fig, axs = plt.subplots(1, 1, figsize = (12,12), subplot_kw={'projection': '3d'}, sharey=True, sharex=True)
        fig.set_tight_layout(tight="tight")

        axs.scatter(self.orients['X'], self.orients['Y'], self.orients['Z'], color='red', marker='o', s=3, label='Orientations')
        axs.scatter(self.interfaces['X'], self.interfaces['Y'], self.interfaces['Z'], color='blue', marker='.', s=4, label='Interfaces')


    def interfacesDF(self):
        return self.interfaces

    def orientsDF(self):
        return self.orients
