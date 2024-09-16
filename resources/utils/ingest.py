import pandas as pd

import logging
import logging.config

class IngestCSV:
    def __init__(self, filepath, known_types=None, range_constraints=None):
        """
        Initialise the CSV ingestion process with the specified file path, known column types, 
        and optional range constraints.

        Parameters:
        - filepath (str): The path to the CSV file.
        - known_types (dictionary): A dictionary of known column types, e.g., {'column_name': 'str', 'age': 'int'}.
        """
        self.logger = logging.getLogger('ingest')
        
        self.filepath = filepath
        self.known_types = known_types or {}
        self.range_constraints = range_constraints or {}

        # placeholder for ingested data
        self.data = None

    def load_csv(self):
        """
        Load the CSV file specified by self.filepath into a pandas DataFrame, applying column types according to the known_types dictionary.
        Once the data has been loaded the function also calls self.fix_data_ranges to impose range limitations.
        
        Returns:
        - pd.DataFrame: Pandas DataFrame containing the loaded data from the CSV file with the specified column types and parsed dates.
        
        Raises:
        - FileNotFoundError: If the file specified by `filepath` does not exist or the file does not exist.
        - pd.errors.EmptyDataError: If the CSV file is empty.
        - pd.errors.ParserError: If there is an issue parsing the CSV file.
        """
        try:
            # identify columns that need to be parsed as dates
            date_columns = [col for col, dtype in self.known_types.items() if dtype == 'datetime64[ns]']
        
            # remove date columns from known_types to prevent conflicts
            non_date_types = {col: dtype for col, dtype in self.known_types.items() if dtype != 'datetime64[ns]'}

            # load CSV file, apply known types to relevant columns, and parse date columns
            self.data = pd.read_csv(self.filepath, dtype=non_date_types, parse_dates=date_columns)
            
            self.logger.info("Raw data loaded")

            if self.range_constraints:
                self.fix_data_ranges()
                
                self.logger.info("Range limitations imposed")
            
            return self.data
        
        except Exception as e:
            self.logger.error(e)
            raise Exception("Error occured while loading CSV")
            exit(1)
        
    
    def fix_data_ranges(self):
        """
        Apply range constraints to specified columns in the DataFrame based on the constraints specified in dictionary: `self.range_constraints`.

        Note:
        - Columns not specified in `self.range_constraints` are not affected by this method.
        - Invalid dates in datetime columns are converted to NaT and do not affect the range constraints.
        
         Returns:
        - None: This method updates the `self.data` DataFrame based on the range constraints.
        """
        
        for column, (min_val, max_val) in self.range_constraints.items():   # iterate through the provided range_constraint dictionary
            if column in self.data: # check if column is present within the loaded CSV
                if pd.api.types.is_string_dtype(self.data[column]):
                    self.logger.warning(f"Warning: Column '{column}' has a string datatype and therefore constraints cannot be applied.")
                else:
                    if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                        self.data[column] = pd.to_datetime(self.data[column], errors='coerce')  
                        
                    if pd.notna(min_val) and pd.notna(max_val):
                        self.data[column] = self.data[column].apply(
                            lambda x: max(min(x, max_val), min_val) if pd.notna(x) else x
                        )
                    else:
                        self.logger.warning(f"Warning: Column '{column}' has null constraints.")
            else:
                self.logger.warning(f"Warning: Column '{column}' not found in the data.")