import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

import pandas as pd

LOGGER = logging.getLogger(__name__)

class SpotifyConnector():
    
    """
    Class to connect and get data from Spotify API
    """
    
    def __init__(self):
        pass
        
    def connect(self, user_id: str, user_code: str):

        """
        A subfunction to be used in data_query
        Args:
            user_id: Spotify Developer ID
            user_code: Spotify Developer Secret CodeID
        Returns:
            connection search object
        """
        
        if not user_id or not user_code:
            raise("User Id or user_code is not being declared")
            
        try:
        
            client_credentials_manager = SpotifyClientCredentials(client_id=user_id, client_secret=user_code)
            sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)    
            return sp
        
        except Error as e:
            
            LOGGER.critical(f"{e}")
            
            return
        
    def data_query(self, 
                   user_id: str,
                   user_code: str,
                   year: int,
                   records_per_query: int = 50,
                   n_track: int = 1000,
                   query_type: str = 'track', 
                   local_save: bool = True) -> pd.DataFrame:
        
        """
        The data collection is divided into 2 parts: the track IDs and the audio features.
        At the moment only maximum 1000 track IDs are collected from the Spotify API.
        
        Limitations:
            limit: a maximum of 50 results can be returned per query
            offset: this is the index of the first result to return, 
            E.g.: so if we want to get the results from 100 forward in the list 
                  set the offset to 100.
        
        Args:
            user_id: Spotify Developer ID
            user_code: Spotify Developer Secret CodeID
            year: Year of tracks being queried
            n_track: Number of tracks to be queried
            records_per_query:  (<= 50) a Limit of 50 results can be rceived per query from Spotìy
            query_type: track = track ID
                        audio = Audio features              
            local_save: locally save to csv if needed 
    
        Returns:
              pd.DataFrame with features
        """
        
        sp = self.connect(user_id, user_code)
        
        if sp is None:
            return
        
        print('Connection Successfully. Start querying....')
        
        # create empty lists where the results are going to be stored
        list_artist_name = []
        list_track_name = []
        list_popularity = []
        list_track_id = []

        for step in tqdm(range(0, n_track, records_per_query)):
            
            track_results = sp.search(q=f'year:{year}',
                                      type=query_type, 
                                      limit=records_per_query,
                                      offset=step)
    
            for _, track_info in enumerate(track_results['tracks']['items']):
                list_artist_name.append(track_info['artists'][0]['name'])
                list_track_name.append(track_info['name'])
                
                list_track_id.append(track_info['id'])
                list_popularity.append(track_info['popularity'])
        
        # Condense those information in dataframe        
        df = pd.DataFrame({'artist_name':list_artist_name,
                           'track_name':list_track_name,
                           'track_id':list_track_id,
                           'popularity':list_popularity})
        
        if local_save:
            df.to_csv('spotify_track_data.csv', index=False)
        
        LOGGER.debug("FINISH QUERYING")

        return df