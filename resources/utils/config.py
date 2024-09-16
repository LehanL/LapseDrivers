import os
import pandas as pd
import logging
import logging.config

from configparser import ConfigParser


class EnvConfig:
    
    def __init__(self, args):
        """
        Loads a command-line arguments and configures application environment parameters.

        Parameters:
        - args (argparse.Namespace object): command-line arguments.

        Returns:
        - None: This function does not return a value but updates the application parameters stored in self.
        """    

        # initialise config parser and let it read the config.ini file
        self.config_object = ConfigParser()
        self.config_object.read(args.config_file)

        # load command-line arguments for the application environment
        self.envir = args.envir
        
        if (self.envir == 'local'):
            # load project parent directory
            parent_directory =  os.getcwd()

            # load logging config
            logging.config.fileConfig(os.path.join(parent_directory,"resources","configs","logging.conf"))

        # setup logger
        logger = logging.getLogger('environment')

        # load command-line arguments as application parameters
        self.train = args.train
        self.validate = args.validate
        self.save_processed_data = args.save_processed_data

        # reset score argument to False if no date was provided or if the date is in the incorrect format
        if (args.score and args.score_date is None):
            
            self.score = False
            
            logger.warning(f"Score argument set to False as no score start date was provided")
        
        elif (args.score and args.score_date is not None):
            
            self.score_date = pd.to_datetime(args.score_date, format='%Y-%m-%d', errors='coerce')
            
            if pd.isna(self.score_date):
                self.score = False

                logger.warning(f"Score argument set to False as the score start date was provided in an incorrect format")

            else:
                self.score = args.score

        # setup file names 
        self.surveyData = self.config_object["sources"]["surveyData"]
        self.lapseData = self.config_object["sources"]["lapseData"]
        self.processedData = self.config_object["fileNames"]["processedData"]
        self.model = self.config_object["fileNames"]["model"]
        self.performanceGraph = self.config_object["fileNames"]["performanceGraph"]
        self.shapGraph = self.config_object["fileNames"]["shapGraph"]
        self.limeReport = self.config_object["fileNames"]["limeReport"]
        self.lapseFeatureImportanceGraph = self.config_object["fileNames"]["lapseFeatureImportanceGraph"]
        self.lapseFeatureImportanceSummary = self.config_object["fileNames"]["lapseFeatureImportanceSummary"]

        # setup paths required by the application
        if (self.envir == 'local'):
            logger.info(f"Project root path is: {parent_directory}")

            self.dataDir = os.path.join(parent_directory,"data","raw")
            self.procDataDir = os.path.join(parent_directory,"data","processed")
            self.modelDir = os.path.join(parent_directory,"models")
            self.reportDir = os.path.join(parent_directory,"reports")
            self.figureDir = os.path.join(parent_directory,"reports","figures")
