# import libraries
import os
import sys
import time
from datetime import datetime
from argparse import ArgumentParser

from resources.utils.config import EnvConfig
from resources.utils.ingest import IngestCSV
from resources.utils.transform import DualSourceTransform
from resources.utils.model import HistGBMPipeline
from resources.utils.report import FeatureImportanceAnalyser

import logging
import logging.config


# define argument parser. 
def build_argparser():
    

    ap = ArgumentParser()
    ap.add_argument("-e",   # a single-letter alias.
        "--envir",          # longer, more descriptive alias.
        type=str,           # specify argument datatype
        default="local",    # defualt argument if no explicit argument was provided at application launch
        help="Name of runtime environment") # Descriptive tp explain purpose of argument
    ap.add_argument("-t", "--train", type=str, default=True, help="True or False: Train Model")
    ap.add_argument("-v", "--validate", type=str, default=False, help="True or False: Validate Model")
    ap.add_argument("-s", "--score", type=str, default=False, help="True or False: Use model to score. Must provide Score Date.")
    ap.add_argument("-sd", "--score_date", type=str, default=None, help="Month Key where Score data starts. Format: yyyy-mm-dd")
    ap.add_argument("-spd", "--save_processed_data", type=str, default=False, help="True or False: Save processed data")

    #load config .ini file
    if (ap.parse_args().envir == "local"):
        file = os.path.join(os.getcwd(),"resources","configs","config.ini")

    # allocate loaded config file to argument
    ap.add_argument("-cf", "--config_file", type=str, default=file, help="Optional configuration file")
    
    logging.info("Argument parser has been setup")

    # return argument parser
    return ap

# define application pipeline. 
class Pipeline:
    def __init__(self, env_prop):
        """
        Initialise primary project pipeline.

        Parameters:
        - env_prop (EnvConfig class instance): class maintaining project configuration parameters.

        Returns:
        - None: This function does not return a value but updates self.env.
        """   

        logging.info("Initialise Pipeline")

        self.env = env_prop

    def run_pipeline(self):
        """
        Primary project pipeline.

        Returns:
        - None: This function does not return a value but initialised classes and executes functions to faciliate the working of the application.
        """   

        logging.info("Running Project Pipeline")

        # check status of train parameter, if argument was set to True model training will commence
        if (self.env.train or self.env.score):
            logging.info("Running data ingestion")
            
            # initialise csv data loader
            survey_loader = IngestCSV(os.path.join(self.env.dataDir,self.env.surveyData), 
                known_types={'POL_NUMBER': 'string', 'MONTH_KEY': 'datetime64[ns]', 'HOW_LIKELY_ARE_YOU_TO_RECOMMEND_THE_PRODUCT': 'Int64', 'GENERAL_FEEDBACK': 'string'},
                range_constraints={'MONTH_KEY': (datetime.strptime('2000-01-01  00:00:00', "%Y-%m-%d %H:%M:%S"),datetime.strptime('2024-08-01  00:00:00', "%Y-%m-%d %H:%M:%S")), 'HOW_LIKELY_ARE_YOU_TO_RECOMMEND_THE_PRODUCT': (1,5)})
            
            # load survey data
            survey_data =survey_loader.load_csv()
            
            # initialise csv data loader
            lapse_loader = IngestCSV(os.path.join(self.env.dataDir,self.env.lapseData), 
                known_types={'POL_NUMBER': 'string', 'MONTH_KEY': 'datetime64[ns]', 'AGE': 'Int64', 'DURATION': 'Int64', 'GENDER': 'string', 'LAPSE_IN_12M': 'bool'},
                range_constraints={'MONTH_KEY': (datetime.strptime('2000-01-01  00:00:00', "%Y-%m-%d %H:%M:%S"),datetime.strptime('2024-08-01  00:00:00', "%Y-%m-%d %H:%M:%S")), 'DURATION': (1,563)})

            # load lapse data
            lapse_data =lapse_loader.load_csv()

            logging.info("Starting Dataframe Transformation")

            # set lapse dataframe parameters 
            keys = ['POL_NUMBER', 'MONTH_KEY']  #Dataframe unique keys
            raw_text = 'GENERAL_FEEDBACK'   # raw string fields (can be a list)
            target =  'LAPSE_IN_12M'    # target column
            segmentor = 'DURATION'

            # initialise data transformation class. Class also accepts the type of merge operation to perform. Accepts 'inner', 'outer', 'left', or 'right'. Defaults to 'inner'.
            lapseTransform = DualSourceTransform(survey_data,lapse_data,keys,target,raw_text) 
            # drop rows with empty keys
            # lapseTransform.drop_empty_keys()
            # drop columns and rows (apart from the set dataframe parameters) that maintain more than the threshold worth of NULLs
            lapseTransform.drop_sparse_data()   # can provide function with a float value to set threshold e.g. 0.75
            # generate features from raw text fields
            lapseTransform.apply_sentiment_analyser()
            # drop raw text fields
            lapseTransform.drop_column(raw_text)
            # impute missing values. Function accepts optional segmentation field to improve numeric imputation.
            lapseTransform.impute(segmentor)
            # standardise and encode categorical fields
            lapseTransform.encoder()
            # scale numeric fields
            lapseTransform.scaler()

            if (self.env.save_processed_data):
                # save processed data if requested
                lapseTransform.store_data(os.path.join(self.env.procDataDir,self.env.processedData))

            if (self.env.score):
                # split off scoring set if requested and month key has been provided. Alter first argument if another field is used for by date splitting
                score_df = lapseTransform.split_by_date('MONTH_KEY', self.env.score_date)
            
            # perform a train and test split 
            x_train, x_test, y_train, y_test, features = lapseTransform.split_data()

            # initialise HistGBM modelling pipeline.
            lapseHistGBM = HistGBMPipeline(x_train, x_test, y_train, y_test, features, target)

            if (self.env.train):
                logging.info("Running training pipeline")
                
                # Optimise and train HistGBM model.
                lapseHistGBM.HistGBM_optimise_and_train()

                # If validation requested compute test set performance and plot metric results
                if (self.env.validate):
                    lapseHistGBM.validate_performance()
                    lapseHistGBM.plot_performance(os.path.join(self.env.figureDir,self.env.performanceGraph))

                # save trained model
                lapseHistGBM.save_model(os.path.join(self.env.modelDir,self.env.model))

            if (self.env.score): 
                # remove keys and target (should be empty) from scoring dataframe
                remove_col = keys + [target]
                x_score = score_df.drop(columns=remove_col)

                # load model   
                lapseHistGBM.load_model(os.path.join(self.env.modelDir,self.env.model))
                # produce event predictions using trained model
                y_pred = lapseHistGBM.score(x_score)
                # extract model
                lapseHistGBMModel = lapseHistGBM.get_model()

                # intialise feature importance analyser class
                lapseFeatureAnalyser = FeatureImportanceAnalyser(lapseHistGBMModel, x_train, x_test, x_score, y_test, y_pred)

                # run feature importance functions. Comment out any function not required.
                lapseFeatureAnalyser.compute_permutation_importance()
                # compute shap values
                lapseFeatureAnalyser.compute_tree_shap_values()
                # generate shap analysis graph for set number of the most important variables. 
                lapseFeatureAnalyser.shap_analysis(os.path.join(self.env.figureDir,self.env.shapGraph), 10)
                # compute LIME importance values
                lapseFeatureAnalyser.compute_lime_importance()
                # generate lime analysis for a sample (choose index of sample to be tested)
                lapseFeatureAnalyser.lime_analysis(0, os.path.join(self.env.figureDir,self.env.limeReport))

                # generate feature importance summary graph
                lapseFeatureAnalyser.generate_graphs(os.path.join(self.env.figureDir,self.env.lapseFeatureImportanceGraph))
                # generate aggeregated feature importance .csv report
                lapseFeatureAnalyser.export_csv(os.path.join(self.env.reportDir,self.env.lapseFeatureImportanceSummary))

                logging.info('Primary pipeline complete')



if __name__ == '__main__':
    # initialise timer to track application run time
    initial_start = time.time()
    
    # initialise argument parser and load provided arguments
    env_args = build_argparser().parse_args()
    # set environment configuration properties. Command-line arguments serve as input parameter.
    env_prop = EnvConfig(env_args)

    logging.info('Application configured')

    # initialise pipeline. Environment properties serve as input parameter.
    pipeline = Pipeline(env_prop)
    # run application pipeline
    pipeline.run_pipeline()

    logging.info("Full run done in --- %s seconds ---" % (time.time() - initial_start))