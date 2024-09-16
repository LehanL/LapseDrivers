# README for the Technical Challenge: Lapse Drivers for Retention Campaigns

Author: Lehan L. Lötter

ID: 9404225043087

Date: 16/09/2024

## Introduction
This project application serves as a predictive modeling pipeline designed to identify and understand the drivers of customer lapses. The outcome of the project is intended to inform retention campaigns.

This README document serves as the model design document and will address each building block of the overall project pipeline, providing a detailed description of the architecture and justification for the choices made.

## Flow Diagram
The project data flow diagram can be accessed via the link below:

https://miro.com/welcomeonboard/dVU5aTRXVDBXWFQ5dXR0cXBGckZjWDFMcVZaT25IZnJ3aHRndW1pcnhjb2VmVFRnVWx0bzBYa0JKQ2l4dnZjOHwzNDU4NzY0NTk5NTA4MDc5OTA3fDI=?share_link_id=335637061305

## Notes
The project is operational, provided that the necessary source data have been loaded into the following directory: LapseDrivers/data/raw/.

Below is an example terminal command to execute the application (ensure that the current working directory is set to the project's root):

```bash:
python main.py -t True -v True -spd True
```

It is important to note that each function scripted for this project includes a Docstring, which provides an overview of the function, its parameters, and its output. Formal documentation can be generated from these Docstrings using a tool like Sphinx, if needed.

In addition, the scripts are fully commented and indicate where potential alterations can be done if required. However, refer to this document as the primary source of guidance for potential alterations. 

## Important Files
Please take note of the following important files in the project directory:

* main.py <- Serves as the primary Python script orchestrating the project flow. 
* requirements.txt <- List of libraries utilised by the project. 
* STRUCTURE.md <- Stored within LapseDrivers/docs/ providing a map of the project's file structure

## Libraries
The following libraries were utilised for the implementation of the project outlined in this document:
* Pandas
* Numpy
* Scikit-learn
* Natural Language Toolkit
* SHAP
* LIME
* Seaborn
* Matplotlib

Run the following command in your terminal to install the libraries listed above. 
```bash:
pip install -r requirements.txt
```

## Report Interpretation
The application, when scoring is requested, generates a .csv file for reporting purposes. Interpretation of the scores is as follows:

* SHAP mean value: Represents the contribution of each feature to the model's prediction for individual samples, averaged across all samples. Larger SHAP values indicate a greater contribution to the overall prediction, but this contribution can be either positive or negative. To prevent positive and negative scores from canceling each other out, the absolute values of the scores are averaged. An additional image, named by default as lapse_HistGBM_shapGraph.png, is generated to show the full range of contributions each feature makes across all scoring samples. This enables the reviewer to assess whether the feature contributes in a single direction or in both.
* LIME importance values: LIME explains how individual features contribute to specific predictions, computed on a per-sample basis. The reported score is the mean of the scores produced across the entire scoring set. Larger LIME values indicate greater significance for the feature. Since LIME is designed for local interpretation and can vary across instances, a report (lapse_HistGBM_LIMEReport.html) is generated to assess feature importance for a specific sample. Note that the report is only for one sample, and the sample ID can be specified. 
* Permuation Importance values:  Permutation importance measures the change in model performance when the values of a feature are randomly shuffled. This tests how important a feature is to the model's accuracy. A larger permutation importance value indicates that the feature is crucial for model performance, as shuffling it significantly decreases accuracy. A near-zero or negative value suggests that the feature is either unimportant or potentially harmful to the model's predictions.

The SHAP and LIME values therefore reveal if a feature is locally important for individual predictions, while permutation importance assess the feature's impact on overall model performance. 

## Phase 1: Application Setup
The image below outlines the design for the setup phase of the application.

![Phase1](docs/flowdiagram/phase1.png)

The Python file main.py is the script that orchestrates the project flow when executed.

The code snippet below is the only portion that runs initially, aside from function definitions, when the file is executed. It invokes a timer to track the execution time of the project and then initialises an argument parser to load the command-line arguments provided when the script is run.

The code passes the provided arguments to the EnvConfig class, which is defined in the config.py file. This class sets up the environment parameters for the application.

Once the application is set up, the project's primary pipeline is initialised with the configured environment properties and then invoked.

```python:
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
```
### Argument Parser
The first part of main.py defines an argument parser, which is the first function called by the script. Using an argument parser allows us to provide commands and arguments to the application, instructing its behavior without modifying the actual script. When deploying the project, it is typically packaged into a Docker image, which can be run in containerised environments such as Kubernetes clusters. The Docker image can include a Bash script to pass necessary arguments to the application when the container starts.

The argument parser defined accepts six unique arguments. Each argument is set up with a default instruction in case no command-line instructions are provided. Each argument is also equipped with a help line to describe to the user what each argument is looking for.
```python:
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
```
### Application Configuration
The EnvConfig class is invoked by main.py to set up the application environment parameters. The class receives the command-line arguments as input.

The EnvConfig class uses configparser to read the provided configuration file, which defaults to config.ini. The config.ini file serves as the primary location for user configuration. It provides the application with relevant file names, including not only the names of the source files the application relies on, but also future file names for images and reports generated by the application. The .ini file is also where user-specific information, such as usernames, firewall keys, etc., can be stored for deployment. While primary categories for lab, dev, and prod have been created, they are empty since, for this project, it is assumed that the application will run in a local environment.

Based on the assumption of local deployment, the project subdirectory paths are hardcoded in EnvConfig, but only after an environment check. To run the project in other environments, a similar check would need to be added for the target environment, followed by the appropriate configuration of the project subdirectory paths for that particular environment. Alternatively relevant paths can also be stored in the .ini file.

Note that in EnvConfig, there is a check to ensure the inclusion and validity of the scoring date provided via the command-line arguments. This parameter is required for scoring to occur. It is assumed that only two sources will be provided and that the scoring set will be separated solely by date. It is expected that the target column for rows after the specified date will be empty.

```python:
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
```
Please find the config.ini file structure below:
```ini:
[local]
master = local[*]

[lab]

[dev]

[prod]

[sources]
surveyData = customer_survey.csv
lapseData = lapse.csv

[fileNames]
processedData = processed_lapse.csv
model = Lapse_HistGBM.pkl
performanceGraph = lapse_HistGBM_testPerformance.png
shapGraph = lapse_HistGBM_shapGraph.png
limeReport = lapse_HistGBM_LIMEReport.html
lapseFeatureImportanceGraph = lapseFeatureImportanceGraph.png
lapseFeatureImportanceSummary = lapseFeatureImportanceSummary.csv
```
### Pipeline Initialisation
Once the application has been configured the primary project pipeline is initialised with the configured environment parameters.
```python:
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
```

## Phase 2: Data Ingestion
The image below outlines the design for the data ingestion phase of the application.

![Phase2](C:/Users/User/Pictures/phase2.png?v=2)

### Pipeline calls CSV loader
Within the primary application pipeline, the first step is data ingestion. However, we first need to check if the application is requested to perform either training or scoring. Training and scoring are the two primary functions of the solution, and for proper completion of the project, at least one of the two needs to be requested. By default, one function is selected, but there is a possibility that the application could be run with both options set to False.

Once the application passes this check, a CSV loader is initialised for both data sources considered by the solution. Options are provided to enforce specific data types and range constraints. Not all fields need to be explicitly defined; if not specified, the loader function will infer the data types.

It is assumed that, where range constraints are required, they are provided. However, outliers can influence model performance, depending on the model choice. In this solution, numeric columns are tested for skewness, and a log transform is applied if deemed necessary. Alternatively, or additionally, a function could be added to detect outliers and either remove or cap them. Due to time constraints, this additional functionality was not included, but the effort required to add it would be minimal.

It should also be noted that the type and range constraints applied were provided within the problem design details. It was assumed that these are the only fields requiring enforcement. However, if additional fields require constraints, a developer can add them to the initialisation of the loaders. Alternatively, the type and range constraints could be configured in the .ini file or within the argument function. It is recommended to use the .ini file, as passing dictionaries as arguments could make application execution more cumbersome. 
```python:
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
```

The following script covers the IngestCSV class stored in ingest.py. This class is initialised separately for each CSV that needs to be loaded. It was opted to require a different class instance per request to protect each load operation so that no configuration interference might take place.

Initialisation requires the source file path and optionally accepts known types and range constraints. However, the last two parameters are not mandatory, as it is expected that more variables will be present than those predefined.

The loading function separates out date columns because pandas treats date fields differently. Although the developer could be required to provide the date fields separately, for simplicity and safeguarding, the decision was made to automatically split these fields.

Note that, in this instance, date fields are assumed to be of type datetime64[ns]. If other datetime formats are expected, the developer can expand the filter with alternative options.

The loader calls the fix_ranges_function if range constraints are provided. The function includes several safety checks to ensure that the field names provided exist within the source file, that the field in question is not a string type, and that the range constraints are not NULL. Once the data passes these safety checks, a lambda function is used to impose the constraints, instead of using pandas' clip method, since the field to be constrained might be a datetime field.


```python:
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
```
## Phase 3: Data Transformation
The image below outlines the design for the data transformation phase of the application.

![Phase3](C:/Users/User/Pictures/phase3.png)

After the CSV data sources have been loaded, they require transformation and cleaning before being applied to the machine learning training pipeline. Data preprocessing is essential for reducing the presence of erroneous data that could mislead the training algorithm, and for minimising noise, which helps reduce the chance of overfitting. Finally, scaling is employed to prevent the model from overweighting numeric features with extreme values.

The code extract below comes from the main pipeline function within main.py. After the source data is loaded, the DualSourceTransform class is initialised, which contains all necessary functions to preprocess the data. The DualSourceTransform class assumes that the training set is split between two sources, with the only shared features being the keys linking the two sets. Before initialising the class, four parameters are defined:

1. keys, representing unique identifiers that link the two source files. 
2. raw_text, an unstructured text field. 
3. target, the dependent variable the model will predict.
4. segmentor, used to segment the data for improving missing value imputation. This field can be set to None as the imputer is designed to function without it.

The sequence of data preprocessing follows the data flow diagram presented for Phase 4. First, the transform class is initialised, followed by the application of a series of preprocessing functions. Note that none of these functions return the source table directly. Instead, the class retains the DataFrame until preprocessing is complete, ensuring consistency in the data transformation process. This design also allows developers to adjust the preprocessing steps, either by changing their order or omitting certain steps.

Note that the encoded data parameters were provided in the problem design details and are assumed to be static. However, if changes to these parameters are needed, a developer can modify them during the parameter assignment step. Alternatively, the parameters can be configured in the .ini file or passed as function arguments. Using the .ini file is recommended, as passing dictionaries as arguments can complicate application execution.

```python:
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
```
### DualSourceTransform Intialisation
The transform class retains the DataFrame until preprocessing is complete, ensuring consistency throughout the data transformation process. Therefore, the initialisation step requests all the necessary parameters to perform the specified preprocessing tasks, except in cases where the preprocessing step requires a threshold or similar criterion for decision-making. Functions that require additional input have default values provided if the parameter is not supplied.

Since the sources are assumed to always consist of two distinct datasets linked by keys, the initialisation step invokes a merge function to join the two sources using the keys as links. By default, the function performs an inner join to ensure data consistency for training. In instances where the survey was completed for only some customers and is expected to serve only as a supplement, the 'how' parameter can be set to perform a left or right join instead.


```python:
class DualSourceTransform:
    def __init__(self, df1, df2, keys, target_col, raw_text, how='inner'):
        """
        Initialise the Transform class to preprocess two source dataframes.

        This constructor sets up the merging operation between two supplied dataframes (`df1` and `df2`).
        The constructor also identifies columns to ignore during subsequent transformations.

        Parameters:
        - df1 (pd.DataFrame): The first source dataframe.
        - df2 (pd.DataFrame): The second source dataframe.
        - keys (str or list of str): The column(s) on which to perform the merge operation.
        - target_col (str): The target column to be used for modeling.
        - raw_text (str or list of str): The column containing raw text data to be used for processing.
        - how (str, optional): The type of merge operation to perform. Accepts 'inner', 'outer', 'left', or 'right'. Defaults to 'inner'.

        Attributes:
        - ignore_columns (set): Set of columns to ignore in transformations, including keys, target column, and raw text column.
        - merged_df (pd.DataFrame): Resulting dataframe after merging `df1` and `df2`.
        """
        self.logger = logging.getLogger('transform')
        
        self.df1 = df1
        self.df2 = df2
        self.keys = [keys] if isinstance(keys, str) else keys
        self.raw_text = [raw_text] if isinstance(raw_text, str) else raw_text
        self.target_col = target_col
        # create ignore columns list to ignore in transformations
        self.ignore_columns = set(self.keys + [self.target_col] + self.raw_text)
        self.how = how       
        
        # call merge dataframes function to join the two source tables
        self.merged_df =  self.merge_dataframes()

        
    def merge_dataframes(self):
        """
        Merge two DataFrames using one or more keys.

        Parameters:
        - df1: First DataFrame.
        - df2: Second DataFrame.
        - keys: A single key (string) or a list of keys (list of strings) for the merge.
        - how: Type of merge to perform ('left', 'right', 'outer', 'inner'). Default is 'inner'.

        Returns:
        - Merged DataFrame.
        """
        merged_df = pd.merge(self.df1, self.df2, how=self.how, on=self.keys)
        
        self.logger.info("Dataframes merged")
        
        return merged_df
```

### Drop Empty Keys

Rows missing any key fields are assumed to be erroneous. To prevent the model from being misled, these rows should be dropped. Note, in the primary pipeline, the drop_empty_keys function is commented out because the source tables are inner joined, making it redundant. If the join type is changed to left or right, it is recommended to reinstate this function.

```python:
    def drop_empty_keys(self):
        """
        Drop rows from a DataFrame where provided key variables (columns) are empty.

        Parameters:
        - self.df: The DataFrame from which to drop rows.
        - self.keys: A single key (str) or a list of keys (list of strings) to check for empty values.

        Returns:
        - DataFrame with rows dropped where any of the key variables are empty.
        """

        # Drop rows where any of the specified key variables are empty
        self.merged_df.dropna(subset=self.keys, inplace=True)
        
        self.logger.info("Rows with empty keys dropped")
```
### Drop sparse data
Rows and columns with a high percentage of missing values (above a certain threshold) are assumed to be erroneous and should be dropped. The default threshold is set at 75%, though it can be adjusted. Too many missing values can cause issues in model performance. For example, in a numeric field, missing values are typically imputed by inferring correct values from available data. However, with too many missing values, this inference cannot be made with sufficient statistical confidence.

In some cases, missing values in categorical fields may hold informative value. For instance, in a column representing qualification status, an empty field might indicate no qualification. Given that qualified individuals are fewer, unqualified individuals would likely be the majority. While this inference may also apply to numerical features, it is less likely, and imputation is more complex.

Due to time constraints, the decision was made to remove all fields with little or no data. However, this is an area that could be reconsidered if model improvement is desired. Implementing a solution would require adding a check to separate numeric and categorical columns, dropping only one type based on the threshold. Alternatively, the imputation function could be expanded to handle numeric columns with many missing values. However, without a deeper understanding of the numeric fields, this approach is not recommended.
```python:
    def drop_sparse_data(self, threshold=0.75):
        """
        Remove rows and columns from the DataFrame where threshold percentage or more of the values are missing.
        
        Parameters:
        - threshold (int): Proportion of missing values to determine if a row/column should be removed.
        
        Return: Alters self.merged_df pandas DataFrame.
        """
        # Calculate the threshold for missing rows within a column
        column_threshold = threshold * len(self.merged_df)
        
        # Identify columns to be dropped based on missing values
        columns_to_drop = [
            col for col in self.merged_df.columns
            if col not in self.ignore_columns and self.merged_df[col].isna().sum() > column_threshold
        ]

        # Remove columns with missing values above the threshold
        self.merged_df.drop(columns=columns_to_drop, inplace=True)

        # Calculate the threshold for missing columns within a row
        row_threshold = threshold * (len(self.merged_df.columns))
        
        # Remove rows with missing values above the threshold
        self.merged_df.dropna(axis=0, thresh=row_threshold, inplace=True)

        self.logger.info("Sparse rows and columns dropped")
```
### Sentiment Analyser
Feature engineering from unstructured text is a vast area open to exploration. Numerous insights can be derived from raw text, each with varying predictive power. Since the survey in this use case focuses on product feedback, a simple feature with high predictive power—sentiment—was chosen. Due to time constraints, the decision was made to engineer a feature that could be created efficiently while maintaining strong predictive performance. Many alternatives exist and could be added as additional features, such as word count, lexical diversity, or adjective count using parts-of-speech tagging. Another common technique is Term Frequency-Inverse Document Frequency (TF-IDF), which measures how closely the words used by a respondent align with those used by others. More advanced, highly predictive techniques include topic modeling or traditional clustering. Topic modeling, using algorithms like Latent Dirichlet Analysis (LDA), identifies common topics within the corpus and assigns each survey the topics present in the feedback. Clustering segments surveys based on similar feedback patterns.

The SentimentAnalyser class was implemented for reusability. The Transform class calls the SentimentAnalyser and passes it the DataFrame and raw text column name. The SentimentAnalyser uses the VADER sentiment lexicon from Scikit-learn, which contains the sentiment polarity of various words. It approximates a sentence's polarity based on the polarity of its individual words. The function apply_sentiment_analyser assumes there could be more than one unstructured text column and thus iterates over self.raw_text.
```python:
    def apply_sentiment_analyser(self):
        """
        Apply sentiment analysis to each column specified in the `raw_text` list.

        Attributes:
        - self.raw_text (list of str): List of column names in the `merged_df` DataFrame that contain raw text data.
        
        Returns:
        - None: This method does not return a value but modifies the `merged_df` DataFrame.
        """
        
        # generate sentiment for each raw text column. 
        for text_col in self.raw_text:
            # initialise sentiment analyser
            sentiment_analyser = SentimentAnalyser(self.merged_df,text_col)
            self.merged_df = sentiment_analyser.apply_sentiment_scores()

            self.logger.info("Sentiment scores calculated and added to dataframe successfully.")
```
The SentimentAnalyser class is provided below. To handle cases where imputation has not yet occurred, the class assigns a neutral sentiment to rows with missing or empty text. It is assumed that no comment equates to neutral sentiment, as customers with strong opinions would likely have added commentary. The class adds the generated sentiment score to the provided DataFrame in a new column called 'sentiment_score'.
```python:
class SentimentAnalyser:
    def __init__(self, df, text_column):
        """
        Initialises the SentimentAnalyser class.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the text data.
        - text_column (str): The name of the column containing raw text data.
        """

        self.df = df
        self.text_column = text_column
        self.sentiment_analyser = SentimentIntensityAnalyzer()  # Initialise VADER sentiment analyser

    def calculate_sentiment(self, text):
        """
        Calculates the sentiment score for a given text using VADER.

        Parameters:
        - text (str): The text to score.

        Returns:
        - float: Sentiment score.
        """
        if pd.isna(text):
            # Return a neutral score for missing text
            return 0.0
        
        # Convert to string if not already
        text = str(text)  
        
        # Use VADER to compute sentiment scores
        sentiment_score = self.sentiment_analyser.polarity_scores(text)
        
        return sentiment_score['compound']  # Return the compound score

    def apply_sentiment_scores(self):
        """
        Applies sentiment scores to the DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with a new column containing sentiment scores.
        """
        if isinstance(self.text_column, str) and self.text_column in self.df.columns:
            # Apply the sentiment score calculation to each row in the DataFrame
            self.df['sentiment_score'] = self.df[self.text_column].apply(self.calculate_sentiment)
        else:
            raise ValueError("text_column must be a valid string column name in the DataFrame.")
        return self.df
```
### Imputation
There are numerous common strategies for handling missing data. As mentioned in the "Drop sparse data" section, missing values often hold informative power, especially for categorical features. Therefore, empty values in categorical fields were assigned to a new category, 'unknown.'

While similar logic applies to numeric features, imputing missing values in these cases is more complex without understanding the nature of the columns. A simple solution can be the creation of a binary variable flagging missing numerical values. A safer option is to impute missing values using the column's mean, median, or mode. However, in large datasets, these values may not accurately reflect the true data. A more accurate approach is to impute by segment if a field exists that can properly segment the data. A clustering solution can also be used to segment the data, but this is more complex and only necessary if no suitable segmentation feature is available.

If the segmentation feature is numerical, the function will bin it. In this case, the binning strategy uses equal population bins via qcut. Alternatively, equal-range cutoffs can be used, or custom bins can be created based on further insight into the segmentation variable. The developer can optimise this choice based on the selected segmentation variable. For this use case, 'DURATION' was chosen as the segmentation variable, as customer behavior is believed to vary between new and long-term customers.

This use case also expects multiple unexplored columns, which might include categorical fields encoded numerically. Therefore, missing numeric values are imputed using the mode, if a most frequent value exists; otherwise, the mean is used.


```python:
    def impute(self, segment_var=None, bins=5):
        """
        Impute missing values in the DataFrame. 
        Categorical columns missing values are replaced with 'unknown'.
        Numeric column missing values are replaced by the column mode or by the mean if there is no mode available. 
        If a `segment_var` is supplied the numeric column imputation is enhanced to replace missing values with the segment's column mode or mean if no mode is available.
        If `segment_var` is numeric, it bins the column and uses these bins for mode-based imputation. 

        Parameters:
        - segment_var (str or None): Column name for segmentation; if None, no segment-based numeric imputation.
        - bins (int): Number of bins for segmentation if `segment_var` is numeric.

        Returns:
        - None: Modifies `self.merged_df`.
        """

        # Bin the segmentation variable if it's numeric
        if segment_var and pd.api.types.is_numeric_dtype(self.merged_df[segment_var]):
            self.merged_df[segment_var + '_binned'] = pd.qcut(self.merged_df[segment_var], q=bins, labels=False, duplicates='drop')
            # default empty segment values with the bottom bin
            self.merged_df[segment_var + '_binned'] = self.merged_df[segment_var + '_binned'].fillna(0)
            # repoint segment_var to the binned variant
            segment_var = segment_var + '_binned'

        elif pd.api.types.is_string_dtype(self.merged_df[segment_var]):
            # if the segment var is 
            self.merged_df.fillna({segment_var: 'unknown'}, inplace=True)

        # Step through dataframe columns
        for col in self.merged_df.columns:
            if col in self.ignore_columns:
                # Ignore keys and target variable
                continue

            if self.merged_df[col].dtype in ['object', 'string', 'category']:
                # Impute categorical columns with 'unknown'
                self.merged_df.fillna({col: 'unknown'}, inplace=True)

            elif self.merged_df[col].dtype in ['int64', 'Int64', 'float64']:
                if segment_var is not None:
                # Impute numeric columns with the mode based on the segmentation variable if it exists
                    if pd.notna(self.merged_df[segment_var]).all():  # Ensure there are no NaNs in binned variable
                            mode_imputation = self.merged_df.groupby(segment_var)[col].apply(
                            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
                            self.merged_df[col] = mode_imputation.reset_index(level=0, drop=True)
                else:
                    # Impute numeric columns with the overall mode or mean if no segment_var is provided
                    mode_value = self.merged_df[col].mode()[0] if not self.merged_df[col].mode().empty else self.merged_df[col].mean()
                    self.merged_df.fillna({col: mode_value}, inplace=True)
        
        # drop binned column if one was created for imputation
        if '_binned' in segment_var:
            self.merged_df.drop(columns=[segment_var], inplace=True)

        self.logger.info("Imputation successful")
```
### Encoding
Numerous encoding strategies exist for handling categorical variables. Common strategies include one-hot encoding, label encoding, and ordinal encoding. Ordinal encoding is used when categories represent a natural ranking or order. One-hot encoding, on the other hand, creates a new binary feature for each category. While effective, one-hot encoding can significantly increase dimensionality, especially with high-cardinality features. In an already sparse dataset, this increase in dimensionality was considered a trade-off worth accepting, given the issues associated with label encoding. Label encoding assigns integers to categories, which can be interpreted by models as having an implicit continuous or ordinal relationship between categories. This approach can limit interpretability and introduce misleading assumptions about the data. Alternative, more memory-efficient options include frequency encoding and target encoding. However, both of these methods also have limitations in terms of interpretability. 

```python:
    def encoder(self): 
        """
        Encodes specified columns in the DataFrame using one-hot encoding.

        Returns:
        - None: Modifies `self.merged_df`.
        """
        # Initialise list to collect columns to be dropped later
        columns_to_drop = []

        # List of columns that were already one-hot encoded
        one_hot_encoded_columns = []

        # Iterate through DataFrame columns
        for col in self.merged_df.columns:
            if col in self.ignore_columns:
                # Skip columns in the ignore list
                continue

            # Check if the column is of string type
            if pd.api.types.is_string_dtype(self.merged_df[col]):
                # Standardise text columns by converting to lowercase
                self.merged_df[col] = self.merged_df[col].str.strip().str.lower()

                # One-hot encode the column
                one_hot = pd.get_dummies(self.merged_df[col], prefix=col)

                # Convert one-hot encoded columns to integers
                one_hot = one_hot.astype(int)
                
                # Drop the original column and add one-hot encoded columns
                self.merged_df = self.merged_df.drop(col, axis=1)
                self.merged_df = pd.concat([self.merged_df, one_hot], axis=1)
            
        
        self.logger.info("One-hot encoding successful") 
```
### Numerical Scaling
Scaling is highly valuable in machine learning, it ensures fair consideration of features and ensures numerical stability.  Min-max scaling was chosen for numerical features in this use case. While standardisation is generally preferred for normally distributed data and is less sensitive to outliers, the distribution of features in this use case is not known due to problem including unexplored features. Min-max scaling also preserves the natural relationships between the data, which enhances interpretability, an important aspect of this solution.

To address the impact of skewed data, log transformation is applied to numeric columns that exceed a certain skewness threshold. This threshold can be adjusted through an argument provided to the function.

Note, the model choice is less susceptible than algorithms such as linear regression and logistic regression to input data scale. Still it can affect the stability of the solution by affecting evenness of the bin distributions reducing model performance. In addition, if down the line A-B testing is required to compare the viability of different models, consistency in data preparation would be required. 
```python:
    def scaler(self, log_transform_threshold=0.5): 
        """
        Apply Min-Max scaling to numeric columns in the DataFrame.
        The function first assesses the numeric columns skewness, if above the set threshold log transformation will be applied before min-max scaling.

        Parameters:
        - log_transform_threshold (float, optional): The threshold for skewness above which log transformation is applied. Default is 0.5.

        Returns:
        - None: Modifies `self.merged_df` in place.
        """
        # Initialise MinMaxScaler
        scaler = MinMaxScaler()

        # Iterate over columns to preprocess
        for col in  self.merged_df.columns:
            if col in self.ignore_columns:
                # Skip columns in the ignore list
                continue
            
            # identify numeric columns
            if pd.api.types.is_numeric_dtype(self.merged_df[col]) and not pd.api.types.is_bool_dtype(self.merged_df[col]):
            # Check skewness to determine if log transformation is needed
                if (self.merged_df[col] >= 0).all() and abs(skew(self.merged_df[col].dropna())) > log_transform_threshold:
                    # Apply log transformation
                    self.merged_df[col] = np.log1p(self.merged_df[col])  # Clip lower bound to avoid log(0)
                    
                # Apply Min-Max Scaling
                self.merged_df[[col]] = scaler.fit_transform(self.merged_df[[col]])

        self.logger.info("Numerical min-max scaling successful")
```

## Phase 4: Modelling ETL
The image below outlines the design for the modelling extract, transform, load (ETL) phase of the application.

![Phase4](C:/Users/User/Pictures/phase4.png?v=2)

The behaviour of the modelling ETL is dependent on the arguments provided to the application at execution. First the primary pipeline checks to see if the processed data frame needs to be saved. Next it checks if scoring is going to take place. Both checks invokes a function if passed. The first step is included to ensure replication. Note, no functionality is currently included to load the data. A future addition can be the addition of another argument to check if data will be loaded, a check can then be added to skip the entire preprocessing step if true, loading the data instead. 

```python:
            if (self.env.save_processed_data):
                # save processed data if requested
                lapseTransform.store_data(os.path.join(self.env.procDataDir,self.env.processedData))

            if (self.env.score):
                # split off scoring set if requested and month key has been provided. Alter first argument if another field is used for by date splitting
                score_df = lapseTransform.split_by_date('MONTH_KEY', self.env.score_date)
            
            # perform a train and test split 
            x_train, x_test, y_train, y_test, features = lapseTransform.split_data()

```
The following code snippet sits within the DualSourceTransform class and is responsible for saving the processed DataFrame to the provided path.
```python:
    def store_data(self, save_path):
        """
        Save the processed data as CSV.

        Parameters:
        - save_path (str): OS path where dataframe will be stored as .csv

        Returns:
        - None: Stores Merged Pandas Dataframe in path provided.
        """
        try:
            # Save the DataFrame as a CSV file
            self.merged_df.to_csv(save_path, index=False)

            self.logger.info(f"Processed data successfully saved to {save_path}")

        except Exception as e:
            self.logger.error(e)
```
### Scoring set extraction
As mentioned in the "Application Configuration" section, the assumption is made that the scoring set will form part of the same source but with later date. Thus, if scoring is requested the scoring set needs to be split off from the training set before initiation of the model training pipeline. Note, the scoring set was split of only after preprocessing ensuring consistency of the data preprocessing. 

```python:
    def split_by_date(self, date_col, split_date):
        """
        Split `self.merged_df` into two DataFrames based on the provided datetime.

        Parameters:
        - date_col (str): The name of the column containing datetime values used for splitting.
        - split_date (datetime-like): The datetime value used to split the DataFrame.

        Returns:
        - score_df (pd.DataFrame): DataFrame with rows on or after `split_date`.
        """
        # Ensure the date column is in datetime format
        self.merged_df[date_col] = pd.to_datetime(self.merged_df[date_col], errors='coerce')

        # Split the DataFrame based on the provided split_date
        score_df = self.merged_df[self.merged_df[date_col] >= split_date]
        self.merged_df = self.merged_df[self.merged_df[date_col] < split_date]
        

        self.logger.info(f"Scoring set split from self.merged_df by date: {split_date}")

        return score_df
```
### Train-Test Split

Before model training can begin, the processed dataset needs to be split into a training set and a test set. Performing model validation on an unseen test set is the best approach to accurately assess model quality, to review any potential overfitting and bias. The function accepts two parameters with default values: test_size, which controls the proportion of data allocated to the test set, and random_state, which sets the seed for the random number generator to ensure replicability. Note that this approach assumes there is sufficient data for an appropriate train-test split. If the dataset is too small, cross-validation would be recommended instead.
```python:
    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Seed used by the random number generator.

        Returns:
        - x_train (pd.DataFrame): Pandas Dataframe for machine learning training.
        - x_test (pd.DataFrame): Pandas Dataframe for model validation.
        - y_train (pd.DataFrame): Pandas Dataframe containing the event field for machine learning training.
        - y_test (pd.DataFrame): Pandas Dataframe containing the event field for model validation.
        - features (list of str): List containing the features considered for machine learning.
        """
        
        features = [col for col in self.merged_df.columns if col not in self.ignore_columns]

        x_train, x_test, y_train, y_test = train_test_split(
            self.merged_df[features], self.merged_df[self.target_col], test_size=test_size, random_state=random_state)
        
        self.logger.info("Train Test split performed successfully")
            
        return  x_train, x_test, y_train, y_test, features
```

## Phase 5: Model Pipeline
The image below outlines the design for the model pipeline.

![Phase5](C:/Users/User/Pictures/phase5.png?v=2)

The model pipeline behaves differently depending on the command-line arguments provided at execution. The HistGBMPipeline class is invoked regardless of whether training, scoring, or both are requested. The HistGBMPipeline class focuses on training a Histogram Gradient Boosting Machine (HistGBM) model and using it for scoring. Once initialised, the training function is called if training is requested. If both training and validation are requested, validation is subsequently performed. Finally, the optimised model is saved to a path specified in config.py using Pickle. The following subsections will explore the model's optimisation, training, and validation processes, explaining the underlying logic.

Note that the HistGBMPipeline class can be replaced by a developer with a similar pipeline for any other desired model training algorithm.

Additionally, the training and testing datasets are provided to the class even if only scoring is requested, as the model's purpose is to report on decision-making and lapse drivers, including generating a report on model weights from the training set. Alternatively, this requirement can be removed by generating the model weights report once after training. The aggregate feature importance report could then be initialised and reloaded during scoring, appending the scores generated from the scoring set. This alternative approach is preferred, as it would improve solution efficiency, but it would require an additional two to four hours of development time due to the required restructuring, excluding the necessary testing.
```python:
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
```
Please see the HistGBMPipeline class initialisation below:
```python:
class HistGBMPipeline:
    def __init__(self, x_train, x_test, y_train, y_test, features, target_col):
        """
        Initialises the HistGBM modeling pipeline.

         Parameters:
        - x_train (pd.DataFrame): The DataFrame containing the training features.
        - x_test (pd.DataFrame): The DataFrame containing the testing features.
        - y_train (pd.Series or pd.DataFrame): The target variable for training data.
        - y_test (pd.Series or pd.DataFrame): The target variable for testing data.
        - features (list or str): A list of feature names to be used in the model. 
          If a single feature name is provided as a string, it will be converted to a list.
        - target_col (str): The name of the column representing the target variable.

        Attributes:
        - model (HistGradientBoostingClassifier): Placeholder for the HistGBM model instance.
        - y_pred (pd.Series or np.ndarray): Placeholder for predicted target values.
        - y_prob (pd.Series or np.ndarray): Placeholder for predicted probabilities (if applicable).

        """
        self.logger = logging.getLogger('model')

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = [features] if isinstance(features, str) else features
        self.target_col = target_col
        
        # Placeholder for model and data
        self.model = None
        self.y_pred = None 
        self.y_prob = None 
        self.score_dict = {}
```
### Histogram GBM optimisation and training
The problem this application aims to solve has several key aspects to consider in the choice of machine learning model. Fundamentally, it is a supervised classification problem. However, the goal is not only to predict lapses but also to identify the main drivers behind those lapses to inform retention campaigns. Therefore, ease of interpretation is crucial. Additionally, due to the nature of survey data, it is expected that the training data will be sparse in a high-dimensional space. Finally, the data includes both numeric and categorical features.

A popular choice for classification problems where interpretability is important and both numerical and categorical data are present is Gradient Boosting Machines (GBM). GBM is a boosting approach that improves performance by using multiple simpler models in succession, specifically decision trees. Each tree is built to correct the errors of the previous one, and the algorithm uses gradient descent to minimise a loss function.

Histogram GBM is a variant of GBM that uses a histogram approach to split the data, which is more efficient for high-dimensional datasets. Instead of considering each value as a potential split point, HistGBM creates bins (histograms) for feature values, grouping them into intervals. This reduces the complexity and computational cost of finding the best split.

Alternatives include logistic regression, which performs poorly in high-dimensional spaces and assumes linearity between inputs and targets, which might not hold in large, complex datasets. Support Vector Machines are computationally expensive, relying on kernel methods to handle non-linearity. Neural Networks, while powerful, are prone to overfitting on sparse datasets, are computationally expensive, and have poor interpretability.

GBM models depend on good optimisation. Hyperparameter optimisation for this problem will be done through random search, which is a simpler and more computationally efficient alternative to grid search. Random search samples random combinations of hyperparameters within specified ranges rather than trying all possible combinations.

The hyperparameters optimised include:

* Learning rate: Controls how aggressively the model's weights are adjusted each iteration. 
* Maximum iterations: Defines the maximum number of boosting trees to fit.
* Maximum leaf nodes: Specifies the maximum number of leaf nodes per tree, controlling tree complexity.
* Minimum leaf nodes: Specifies the minimum number of samples required at a leaf node to prevent overfitting.
* Maximum depth: Determines the maximum depth of the trees.
* Ridge regularisation: Adds a penalty proportional to the square of the magnitude of coefficients to the loss function.
* Validation fraction: Specifies the fraction of training data to set aside for validation during training.
* Early stopping: Stops training early if no improvement is seen in the validation set performance for a certain number of iterations, helping to prevent overfitting and saving resources.
* Scoring: Defines the loss function. ROC AUC (Receiver Operating Characteristic Area Under Curve) measures the model's ability to distinguish between classes.

Random search has its own additional parameters:

* Number of iterations: Number of unique hyperparameter combinations to sample.
* cv: Sets the number of folds for cross-validation.
* Random state: Sets the random number generator seed to ensure reproducibility.
* n jobs: Number of CPU cores to use for parallel processing.

Bayesian optimisation is a more advanced and often more efficient method than random search, especially for expensive functions. It models the hyperparameter space with a probabilistic model and uses it to find the best parameters. Each iteration uses a Gaussian process to predict the quality of different parameter sets based on previous results. Although Bayesian optimisation is recommended due to its efficiency, client requirements restrict machine learning to Scikit-learn libraries. A custom implementation of Bayesian optimisation is estimated to require 4-6 hours, including testing, whereas using libraries would take approximately 1 hour.

The code snippet below is a function from the HistGBMPipeline class. It defines the base model, the hyperparameter search range, and the random search function, and then performs optimisation. The RandomizedSearchCV produces a model trained with the best-performing parameters. However, because early stopping may be enforced, the model might not be fully trained by the time training stops, which disables the capability to produce scoring probabilities, a critical feature for desired interpretation capabilities. Therefore, it is decided to retrain the final model with early stopping disabled once the best-performing hyperparameters have been identified.
```python:
    def HistGBM_optimise_and_train(self):
        """
        Optimises hyperparameters for the HistGradientBoostingClassifier using RandomizedSearchCV 
        and trains the model with the best parameters.

        Returns:
        - None: Allocates HistGBM model trained on the optimum hyperparameters to self.model
        """
        # Define the base model
        base_model = HistGradientBoostingClassifier()

        # Define the parameter grid for RandomizedSearchCV
        param_distributions = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_iter': [100, 200, 300, 400, 500],
            'max_leaf_nodes': [15, 31, 63, 127],
            'max_depth': [3, 5, 7, 9, None],
            'min_samples_leaf': [10, 20, 30, 40, 50],
            'l2_regularization': [0.0, 0.1, 0.5, 1.0],
            'early_stopping': [True],
            'scoring': ['roc_auc'],
            'validation_fraction': [0.1, 0.15, 0.2]
        }

        # Setup the RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=50,  # Number of iterations to perform
            scoring='roc_auc',  # Optimise based on AUC score
            cv=5,  # 5-fold cross-validation
            verbose=1,  # Set to 1 or higher to get updates
            random_state=42,  # Seed for reproducibility
            n_jobs=-1  # Use all available cores
        )

        # Fit the RandomizedSearchCV
        random_search.fit(self.x_train, self.y_train)

        self.logger.info("Randomised search hyperparameter optimisation complete")

        # Extract the best model
        best_params = random_search.best_params_
        best_params['early_stopping'] = False  # Disable early stopping for final model

        # Reinitialise and fit the final model on the full training data with best parameters
        self.model = HistGradientBoostingClassifier(**best_params)
        self.model.fit(self.x_train, self.y_train)

        self.logger.info(f"HistGBM model has been refit with the best hyperparameters found: {random_search.best_params_}")
```
### Validation
To evaluate the trained model's performance, it is applied to an unseen test set. Multiple metrics are produced to fully assess the quality of the trained model:

* Accuracy - The proportion of correctly predicted instances out of the total instances.
* Precision - The proportion of true positive predictions out of all positive predictions made by the model.
* Recall - The proportion of true positive predictions out of all actual positives in the dataset.
* Specificity - The proportion of true negative predictions out of all actual negatives in the dataset.
* AUC Score - The Area Under the Receiver Operating Characteristic Curve, which measures the model's ability to distinguish between classes.
* Gini Score - A metric derived from the AUC score that quantifies the inequality among the predicted probabilities for different classes.


The performance metrics are then stored in a dictionary. Once the performance scores have been computed, a graphing function is called from the primary pipeline to plot the metrics.
```python:
    def validate_performance(self):
        """
        Validates the performance of the trained model by computing various metrics.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, specificity, 
                AUC (Area Under the ROC Curve), and Gini coefficient.

        Raises:
            ValueError: If the model has not been trained.
        """
        # Check if model has been trained
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # use model to predict target for the test set
        self.y_pred = self.model.predict(self.x_test)
        
        # use model to predict target probabilities for the test set
        self.y_prob = self.model.predict_proba(self.x_test)[:, 1]

        # Compute performance metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()

        # Compute specificity
        specificity = tn / (tn + fp)

        # Compute auc score and gini
        auc_score = roc_auc_score(self.y_test, self.y_prob)
        gini_score = 2 * auc_score - 1

        # compile performance metrics
        self.score_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity':specificity,
            'auc': auc_score,
            'gini': gini_score
        }

        self.logger.info(f"HistGBM model validation performance: {self.score_dict}")


    def plot_performance(self, storage_path=None):
        """
        Plots performance metrics. Results are stored to the provided storage_path.
        """
        # run validate_perfomance if performance metrics have not yet been generated.
        if not self.score_dict:
            self.score_dict = self.validate_performance()

        # initialise performance plotter
        plotter = ModelPerformancePlotter(self.y_pred, self.y_prob, self.y_test, self.score_dict)
        # plot performance metrics providing it the storage path
        plotter.plot_performance(storage_path)

        self.logger.info(f"Graph plotted and stored in {storage_path}")
```
### Performance Plotting
After the model has been trained and the relevant performance metrics have been computed, the ModelPerformancePlotter is initialised and used to graph the computed metrics using the Seaborn and Matplotlib libraries. The function plots the metrics on a histogram, the ROC curve on another subplot, and the CAP curve on the final subplot. If a path is provided, the generated image is saved.
```python:
class ModelPerformancePlotter:
    def __init__(self, y_pred, y_prob, y_actual, metrics_dict):
        """
        Initialises the ModelPerformancePlotter class.

        Parameters:
        - y_pred (array-like): Predicted labels.
        - y_actual (array-like): Actual labels.
        - y_prob (array-like): Predicted probabilities.
        - metrics_dict (dict): Dictionary containing performance metrics.
        """
        self.y_pred = y_pred
        self.y_actual = y_actual
        self.y_prob = y_prob
        self.metrics_dict = metrics_dict


    def plot_performance(self, storage_path=None):
        """
        Plots performance metrics and saves the figure to the specified path.

        Parameters:
        - save_path (str, optional): Path to save the plot figure.
        """
        # Extract metrics and scores from the metrics dictionary
        metrics = list(self.metrics_dict.keys())
        scores = list(self.metrics_dict.values())
        
        # place extracted metrics in dataframe
        df_performance = pd.DataFrame({
            'Metric': metrics,
            'Score': scores
        })

        # Initialise figure
        plt.figure(figsize=(18, 6))
        
        # Subplot 1: Performance Metrics Bar Plot
        plt.subplot(1, 3, 1)
        sns.barplot(x='Metric', y='Score', data=df_performance)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.xlabel('Metric')
        plt.ylabel('Score')

        # Subplot 2: ROC Curve
        fpr, tpr, _ = roc_curve(self.y_actual, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Subplot 3: CAP Curve
        sorted_indices = np.argsort(self.y_prob)[::-1]
        sorted_y = np.array(self.y_actual)[sorted_indices]
        total = len(sorted_y)
        cum_pos_rate = np.cumsum(sorted_y) / sum(sorted_y)  # Cumulative positive rate
        x = np.arange(1, total + 1) / total  # Percentage of total population
        
        plt.subplot(1, 3, 3)
        plt.plot(x, cum_pos_rate, color='blue', lw=2, label='CAP curve')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        
        # Draw the "perfect model" line
        plt.plot([0, sum(sorted_y) / total, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect Model')

        plt.xlabel('Proportion of Samples')
        plt.ylabel('Cumulative Positive Rate')
        plt.title('Cumulative Accuracy Profile (CAP) Curve')
        plt.legend(loc='lower right')

        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided otherwise show plot
        if storage_path:
            plt.savefig(storage_path)
        else:
            plt.show()
```
Finally the trained model is stored in the provided path. 
```python:
    def save_model(self, filepath):
        """
        Saves the trained model to a file using pickle.

        Parameters:
        - filepath (str): The path where the model should be saved.
        """
        # check if model exists
        if self.model is None:
            self.logger.warning("No model is trained to save.")
            return
        
        # save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        self.logger.info(f"Model saved to {filepath}.")
        

    def load_model(self, filepath):
        """
        Loads a model from a file using pickle and assigns it to self.model.

        Parameters:
        - filepath (str): The path from where the model should be loaded.
        """
        # load model from provided path 
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from {filepath}.")

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            self.logger.error(e)
```
## Phase 6: Scoring
The image below outlines the design for the scoring phase of the application.

![Phase6](C:/Users/User/Pictures/phase6.png?v=2)

If scoring is requested the FeatureImportanceAnalyser class is initialised. The class receives the loaded model, the training set, the testing set, the scoring set, as well as the predicted target. Once initialised multiple feature importance functions can be called. As was the case for data preprocessing, the developer is enabled to alter the functions called. The final export function compiles the generated feature importance scores and produces a .csv file for reporting purposes. The code snippets below will explore the individual functions and explain the insight each will provide. 
```python:
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
```
Before initialising the FeatureImportanceAnalyser class the loaded model needs to be used to predict the target for the scoring set. Note, the scoring function has the capability to return the predicted probability when required for specific feature interpretation methods.
```python:
    def score(self, x_score, return_proba=False):
        """
        Scores the provided dataset using the trained model.

        Args:
            x_score (pd.DataFrame or np.ndarray): Dataframe to be scoring.
            return_proba (bool): If True, returns probabilities; otherwise, returns predicted labels.

        Returns:
            np.ndarray: Predicted labels or probabilities for the input dataset.

        Raises:
            ValueError: If the model has not been trained.
        """
        # check if model exists
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Predict labels or probabilities
        if return_proba:
            y_scored = self.model.predict_proba(x_score)[:, 1]  # Probability for the positive class
        else:
            y_scored = self.model.predict(x_score)

        return y_scored
```
Permutation importance is a technique used to assess the importance of features by measuring the impact of shuffling a feature's values on the model's performance.  Permutation importance reflects how critical a feature is to the model’s predictions by quantifying the loss of predictive power when the feature’s information is disrupted. Permutation importance is considered the standard approach to assess feature importance for tree-based models.

The permutation importance accepts certain parameters. 

* n repeats = Defines the number of times the permutation process is repeated to get a stable estimate of feature importance. 
* random state = The seed for the random number generator, ensures repeatability.

The function compute_permutation_importance, the function stores the average importance score for each feature across all permutations. 

Note, permutation importance is typically measured on the test set in order to perform an unbiased, generalised assessment. 
```python:
    def compute_permutation_importance(self):
        """Compute and log permutation importance."""
        result = permutation_importance(self.model, self.x_test, self.y_test, n_repeats=10, random_state=42)
        self.permutation_importance = result.importances_mean
        self.logger.info(f"Permutation Importance: {self.permutation_importance}")
```
Shapeley Additive exPlanations (SHAP) are based on cooperative game theory. SHAP values show how much each feature contributes to a particular prediction compared to the average prediction. For each feature, SHAP measures how much including it changes the prediction from the baseline. This process is repeated for all possible feature orders to counteract feature relations. SHAP values are also model agnostic, enabling fair comparison between different modeling methodologies. 

For tree based models SHAP values can be produced more efficiently using treeSHAP. Gain is calculated by looking at the difference in the objective function before and after adding a feature into the tree’s splits. It reflects how much each feature helps in splitting nodes to improve the predictive accuracy.

The SHAP explainer is applied to the scoring set, and if required can explain predictions per sample. 

In the function compute_SHAP_values, the function stores the average SHAP values for each feature across all samples in the scoring set.  

```python:
    def compute_tree_shap_values(self):
        """Compute and log Tree SHAP values."""
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer(self.x_score)
        self.shap_agg_values = np.abs(self.shap_values.values).mean(axis=0)
        self.logger.info(f"Tree SHAP Values: {self.shap_values}")
```
Local Interpetable Model-agnostic Explanations (LIME). The gain measures the importance of each feature based on its contribution to the local prediction for a specific instance. LIME creates a simpler, interpretable model (like linear regression) that approximates the behavior of the complex model around a specific data point. The gain importance reflects how much each feature affects the prediction made by this local model. It offers a clear view of feature impacts without needing to interpret the global model.

In the function compute_LIME_importance, the function stores the average LIME gain values for each feature across all samples in the scoring set.  
```python:
    def compute_lime_importance(self):
        """Compute and log LIME importance."""
        explainer = lime.lime_tabular.LimeTabularExplainer(self.x_score.values, mode='classification', 
                                                           feature_names=self.x_score.columns.tolist(),
                                                           class_names=[str(c) for c in np.unique(self.y_pred)], 
                                                           # verbose=True, #Uncomment line if needed to debug lime operation
                                                           discretize_continuous=True)
        lime_importances = []
        for i in range(len(self.x_score)):
            explanation = explainer.explain_instance(self.x_score.iloc[i].values, self.model.predict_proba, num_features=len(self.x_score.columns))
            importance_dict = dict(explanation.as_list())
            lime_importances.append([importance_dict.get(feat, 0) for feat in self.x_score.columns])
        self.lime_importance = np.mean(lime_importances, axis=0)
        
        self.logger.info(f"LIME Importance: {self.lime_importance}")
```
The export_csv function compiles all generated feature importance metrics into a single .csv file and stores it in the path provided. The function populates feature importance scores that were not computed with NaN (Not a Number). The file created can be used for dashboarding or reporting of risk drivers to management.
```python:
    def export_csv(self, save_path='.'):
        """
        Export all importance scores to a CSV file.
        
        Parameters:
        - save_path (str): Path to save the feature importance summary .csv.
        """
        # create Pandas dataframe with all of the aggregated feature importance values
        importance_df = pd.DataFrame({
            'Feature': self.x_train.columns,
            'Permutation Importance': self.permutation_importance if self.permutation_importance is not None else np.nan,
            'Tree SHAP Values': self.shap_agg_values if self.shap_agg_values is not None else np.nan,
            'LIME Importance': self.lime_importance if self.lime_importance is not None else np.nan
        })

        # store dataframe in a .csv
        importance_df.to_csv(f"{save_path}", index=False)

        self.logger.info(f"CSV file saved to {save_path}")
```
## Feature Importance Visualisations
Finally three functions were scripted to visualise the generated feature scores using Matplotlib and Seaborn. The first function generates three subplots to visualise the feature aggregated feature importance scores for three analysis approaches considered.

```python:
    def generate_graphs(self, save_path='.'):
        """
        Generate and save graphs of feature importance scores.
        
        Parameters:
        - save_path (str): Path to save the feature importance graphs.
        """
        # initialise figure
        plt.figure(figsize=(16, 10))

        # Permutation Importance
        if self.permutation_importance is not None:
            plt.subplot(1, 3, 1)
            sns.barplot(x=self.permutation_importance, y=self.x_test.columns)
            plt.ylabel('Features')
            plt.title('Permutation Importance')

        # Tree SHAP Values
        if self.shap_values is not None:
            plt.subplot(1, 3, 2)
            sns.barplot(x=self.shap_agg_values, y=self.x_score.columns)
            plt.ylabel('Features')
            plt.title('Tree SHAP Values')

        # LIME Importance
        if self.lime_importance is not None:
            plt.subplot(1, 3, 3)
            sns.barplot(x=self.lime_importance, y=self.x_score.columns)
            plt.ylabel('Features')
            plt.title('LIME Importance')

        # set graph layout
        plt.tight_layout()
        # save figure
        plt.savefig(f"{save_path}")
        # show figure
        plt.show()

        self.logger.info(f"Graphs saved to {save_path}")
```
The second function produces a graph to display the full range of SHAP values produced by each feature when considering the scoring set. This view provides us with a better understanding on the impact of each feature, it allows us to consider if a feature contributes massively in both the positive and the negative direction, or only in one. It also enables the reviewer to see the variance of scores produced by each feature.
```python:
    def shap_analysis(self, shap_path='.', max_display = 10):
        """
        Plot Tree SHAP values.
        
        Parameters:
        - shap_path (str): Path to save the SHAP summary plot.
        """
        plt.figure()
        shap.summary_plot(self.shap_values, self.x_score, max_display=max_display)
        plt.savefig(f"{shap_path}")
        plt.close()

        self.logger.info(f"Shap analysis saved to: {shap_path}")
```
Lastly, a function was created to analyse the feature importance behaviour on a per sample level. The function receives an index defining which sample within the scoring set to assess. The function produces a full LIME analysis explanation file, storing it as a .html file. 
```python:
    def lime_analysis(self, index=0, lime_path='.'):
        """
        Generate LIME analyses for a given index and save the results.

        Parameters:
        - index (int): The index of the instance to analyse.
        - lime_path (str): Path to save the LIME .html file.
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(self.x_score.values, mode='classification', 
                                                           feature_names=self.x_score.columns.tolist(),
                                                           class_names=[str(c) for c in np.unique(self.y_pred)], 
                                                           # verbose=True, #Uncomment line if needed to debug lime operation
                                                           discretize_continuous=True)

        lime_exp = explainer.explain_instance(
            data_row=self.x_score.iloc[index].values,
            predict_fn=self.model.predict_proba
        )
        
        # Save LIME explanation to an HTML file
        lime_exp.save_to_file(lime_path)

        self.logger.info(f"Lime analysis of sample {index} saved to: {lime_path}")
```
