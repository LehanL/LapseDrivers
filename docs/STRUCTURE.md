LapseDrivers/
│
├── data/          
|   ├── processed/         # storage location for preprocessed base
│   └── raw/               # Location for source data
│
├── data/                  # Data directory (if applicable)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
│
├── DevNotebooks/          
│   └── LapseDriver.ipynb  # Jupyter notebook for project testing
|
├── models/                # Pickled model storage      
|
├── reports/               # Storage for reports created by application
│   └── figures/           # Storage for figures created by application
|
├── docs/                  # Documentation
│   ├── README.md          # Detailed project breakdown
│   └── STRUCTURE.md       # Map of project file structe
│
├── resources/             
|   ├── configs/
|   |   ├── config.ini     # Application configuration file. Alter if output file names need to be changed.
│   |   └── logging.conf   # Logger configuration
│   └── utils/             # Modular machine learning classes utilized by main.py
|       ├── config.py      # Configures application environment parameters.
|       ├── ingest.py      # Holds the IngestCSV class
|       ├── model.py       # Holds the HistGBMPipeline class
|       ├── report.py      # Holds the FeatureImportanceAnalyser class
|       ├── sentiment.py   # Holds the SentimentAnalyser class
|       ├── transform.py   # Holds the DualSourceTransform class.
│       └── visualise.py   # Holds the ModelPerformancePlotter class
│
├── .gitignore             # Git ignore file
├── requirements.txt       # List of dependencies
├── pyproject.toml         # Project metadata and build configuration
├── main.py                # Main script to run the project
└── LICENSE                # License information (e.g., MIT License)