from util.PyplotDiagram import PyplotDiagram
from util.console import console
from util.helper import prepareEnv, cleanEnv
from train.single import *
from train.pre_processing import *
from train.tool import getFeaturesAndLabel

from keras.layers import Dense as KerasDenseLayer
from pandas import read_csv as readCSV

# Use VSCode to open the entire folder, then run this script
# Otherwise, the import may not be solved


def main():
    # Do preparation jobs, like install tools, download datasets to specified path (datasets directory)
    # Do not worry, almost all config would be clean-up if you want
    prepareEnv()
    
    ### Data Cleaning #################################################################################################

    console.info('Data Cleaning: Start')

    # Read Datasets
    console.info('Data Cleaning: 1. Reading datasets')
    df_2015 = readCSV('./datasets/data_2015.csv')
    df_2020 = readCSV('./datasets/data_2020.csv')

    # Data Unification
    console.info('Data Cleaning: 2. Data Unification')
    unifyColNames(df_2015, df_2020)
    unifyColOrder(df_2015, df_2020)
    classToDigitReplace(df_2020, verbose=True)
    
    # Save the modified datasets to .csv files
    console.info('Data Cleaning: 3. Saveing modified datasets')
    df_2015.to_csv('datasets/data_2015_modified.csv', index=False)
    df_2020.to_csv('datasets/data_2020_modified.csv', index=False)

    console.info('Data Cleaning: Complete')
    console.wait(5)
    console.clear()

    ### Train a model for the 2015 dataset ############################################################################

    console.info('Model Training 2015: Start')

    # Feature Selection
    # Select the most influential features (first 90%) to the target value
    console.info('Data Cleaning: 1. Feature Selection (30-40s)')
    # Uses the mutual_info_classif method provided by Keras, which is encapsulated in the selectFeatures method (self-implemented)
    features_selected = selectFeatures(df_2015, labelColName="HeartDisease", threshold=0.9)
    print("{} features selected".format(len(features_selected)))
    print(features_selected)
    console.wait(3)
    # df_2015_fs: the 2015 dataset containing only the selected features
    df_2015_fs = df_2015[features_selected]
    # Save the dataset with only selected features
    df_2015_fs.to_csv('datasets/data_2015_fs.csv', index=False)
    console.info('Model Training 2015: 1. data_2015 with only selected features was saved to model dir')
    
    # Train the model according to 2015 data
    console.info('Model Training 2015: 2. Train binary-classification model for the 2015 dataset')
    # Split train and test set
    x = df_2015_fs.values
    y = df_2015["HeartDisease"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
    # Prepare layer set
    # Input Layer: Dense 16 relu
    # Hidden Layer: Dense 64 relu
    # Hidden Layer: Dense 32 relu
    # Output Layer: Dense 1 sigmoid
    layers = [
        KerasDenseLayer(16, activation="relu", input_dim=len(features_selected)),  # Input Layer (input_dim=15)
        KerasDenseLayer(64, activation="relu"),                                 # Hidden Layer #1
        KerasDenseLayer(32, activation="relu"),                                 # Hidden Layer #2
        KerasDenseLayer(1, activation="sigmoid")                                # Output Layer
    ]
    # From the experiment, 4 epoches is found to be the best trainning configuration
    # On such small dataset use CPU to compute usually have higher efficiency
    # Check Project_Interactive_Demo.ipynb for more information about this configureation 
    # (Train a model for the 2015 dataset/Test Trainning: Find the best configuration)
    model_2015 = getModelByXYColumn(
        y_train, x_train,
        use_CPU=True, layers=layers,
        fit_epoch=4, compile_loss_function='binary_crossentropy', 
        fit_callbacks=[], validation_split=0
    )
    
    # Evaluate the trained model
    console.info('Model Training 2015: 3. Evaluate the trained model')
    result_2015 = testModel(model_2015, y_test, x_test, use_CPU=True)
    
    # Save the trained model
    console.info('Model Training 2015: 4. Saving the trained model')
    if not os.path.exists('models'): os.mkdir('models')
    model_2015.model.save('models/model_2015.h5')

    console.info('Model Training 2015: Complete')
    console.wait(5)
    console.clear()

    ### Make up missing values in the 2020 dataset ####################################################################

    console.info('Missing Values Making Up: Start')

    # Prepare the dataset output directory
    if not os.path.exists('datasets/makedUpDatasets'): os.mkdir('datasets/makedUpDatasets')

    # Default Value Filling 
    console.info('Missing Values Making Up: 1. Default Value Filling')
    # Use special values like -1 and NaN to sign the missing values
    # -1 is used as the symbol of missing values in this project
    df_2020_default = df_2020.copy(deep=False)
    makeUpAllMissingValue(df_src=df_2015, df_dist=df_2020_default, makeUpFunc=defaultValueMakeUp)
    # Save the maked up dataset
    console.info('Missing Values Making Up: 1. Saving dataset maked up with default value filling')
    df_2020_default.to_csv('datasets/makedUpDatasets/data_2020_default.csv', index=False)

    # Average Value Filling
    console.info('Missing Values Making Up: 2. Average Value Filling')
    # Use the average value of each feature to fill the missing values.  
    # In this case, the average values are the most frequent appearing values of each feature
    # because the featues that needed to be maked up are all digit represented classes.
    df_2020_average = df_2020.copy(deep=False)
    makeUpAllMissingValue(df_src=df_2015, df_dist=df_2020_average, makeUpFunc=averageValueMakeUp)
    # Save the maked up dataset
    console.info('Missing Values Making Up: 2. Saving dataset maked up with average value filling')
    df_2020_average.to_csv('datasets/makedUpDatasets/data_2020_average.csv', index=False)

    # ML Model Prediction
    console.info('Missing Values Making Up: 3. ML Model Prediction')
    # This method uses the features shared between the two datasets to build ML models 
    # which will then be used to predict the missing values in the 2020 dataset.
    #
    # The self-defined mlPredictValueMakeUp function will do the model training and data prediction work here.  
    # In general, a model will be created and trained for every feature that needed to be maked up.
    #
    # Every model trained to predict missing values will be saved to 'models/mlModelPredictionMakeUp'
    # with file names 'model_predict_<name of the missing feature>.h5
    #
    # Prepare the ML model output directory
    output_dir='models/mlModelPredictionMakeUp'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    #
    # In training the model for each missing features, a universal set of layers is used to train the models as the following shows:
    # - Input Layer: Dense 16 relu
    # - Hidden Layer: Dense 16 relu
    # - Output Layer: Dense * softmax (* may change when predicting different features) 
    df_2020_ml = df_2020.copy(deep=False)
    makeUpAllMissingValue(df_src=df_2015, df_dist=df_2020_ml, makeUpFunc=mlPredictValueMakeUp, output_dir=output_dir)
    # Save the maked up dataset
    console.info('Missing Values Making Up: 3. Saving dataset maked up with ML model prediction')
    df_2020_ml.to_csv('datasets/makedUpDatasets/data_2020_ml.csv', index=False)

    console.info('Missing Values Making Up: Complete')
    console.wait(5)
    console.clear()

    ### Test performance of the maked up data #########################################################################
    
    console.info('Maked Up Dataset Performance Test: Start')

    # Default Value Filling Method
    console.info('Maked Up Dataset Performance Test: 1. Default Value Filling Method')
    # Split features and label
    x, y = getFeaturesAndLabel(df_2020_default, features_selected, 'HeartDisease')
    # Evaluate the performance of the maked up data
    result_default = testModel(model_2015, y, x, use_CPU=True)

    # Average Value Filling Method
    console.info('Maked Up Dataset Performance Test: 2. Average Value Filling Method')
    # Split features and label
    x, y = getFeaturesAndLabel(df_2020_average, features_selected, 'HeartDisease')
    # Evaluate the performance of the maked up data
    result_average = testModel(model_2015, y, x, use_CPU=True)

    # ML Model Prediction Method
    console.info('Maked Up Dataset Performance Test: 3. ML Model Prediction Method')
    # Split features and label
    x, y = getFeaturesAndLabel(df_2020_ml, features_selected, 'HeartDisease')
    # Evaluate the performance of the maked up data
    result_ml = testModel(model_2015, y, x, use_CPU=True)

    console.info('Maked Up Dataset Performance Test: Complete')
    console.wait(5)
    console.clear()

    ### Train a new model for the 2020 dataset ########################################################################

    console.info('Model Training 2020: Start')

    # Feature Selection
    # Select the most influential features (first 90%) to the target value
    console.info('Model Training 2020: 1. Feature Selection (30-40s)')
    # Uses the mutual_info_classif method provided by Keras, which is encapsulated in the selectFeatures method (self-implemented)
    features_selected = selectFeatures(df_2020, labelColName="HeartDisease", threshold=0.9)
    print("{} features selected".format(len(features_selected)))
    print(features_selected)
    console.wait(3)
    # df_2020_fs: the 2020 dataset containing only the selected features
    df_2020_fs = df_2020[features_selected]
    # Save the dataset with only selected features
    df_2020_fs.to_csv('datasets/data_2020_fs.csv', index=False)
    console.info('Model Training 2020: 1. data_2020 with only selected features was saved to model dir')
    
    # Train the model according to 2020 data
    console.info('Model Training 2020: 2. Train binary-classification model for the 2020 dataset')
    # Split train and test set
    x = df_2020_fs.values
    y = df_2020["HeartDisease"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
    # Prepare layer set
    # Input Layer: Dense 16 relu
    # Hidden Layer: Dense 64 relu
    # Hidden Layer: Dense 32 relu
    # Output Layer: Dense 1 sigmoid
    layers = [
        KerasDenseLayer(16, activation="relu", input_dim=len(features_selected)),  # Input Layer (input_dim=11)
        KerasDenseLayer(64, activation="relu"),                                 # Hidden Layer #1
        KerasDenseLayer(32, activation="relu"),                                 # Hidden Layer #2
        KerasDenseLayer(1, activation="sigmoid")                                # Output Layer
    ]
    # From the experiment, 3 epoches is found to be the best trainning configuration
    # On such small dataset use CPU to compute usually have higher efficiency
    # Check Project_Interactive_Demo.ipynb for more information about this configureation 
    # (Train a model for the 2020 dataset/Test Trainning: Find the best configuration)
    model_2020 = getModelByXYColumn(
        y_train, x_train,
        use_CPU=True, layers=layers,
        fit_epoch=3, compile_loss_function='binary_crossentropy', 
        fit_callbacks=[], validation_split=0
    )
    
    # Evaluate the trained model
    console.info('Model Training 2020: 3. Evaluate the trained model')
    result_2020 = testModel(model_2020, y_test, x_test, use_CPU=True)
    
    # Save the trained model
    console.info('Model Training 2020: 4. Saving the trained model')
    model_2020.model.save('models/model_2020.h5')

    console.info('Model Training 2020: Complete')
    console.wait(5)
    console.clear()

    ### Result Analysis (ploting) #####################################################################################
    exit(0)

    # Train the model according to 2020 data
    df_2020 = readCSV("./datasets/data_2020.csv")
    unifyColNames(df_2020)
    classToDigitReplace(df_2020, getClassToDigitDict())
    model_2020 = getModel(
        df_2020, "HeartDisease",
        use_CPU=True, descr_convert=getClassToDigitDict(),
        layers=[KerasDenseLayer(10, activation="relu")],
        fit_epoch=10
    )
    model_2020_result = testModel(model_2020, use_CPU=True)

    # Try to use previous model by filling up missing value
    df_2015 = readCSV("./datasets/data_2015.csv")

    # Do feature selection to choose the 90% most important columns
    df_2015_selected = getFeatureSelectedDataFrame(df_2015, "HeartDisease", 0.9)

    # Draw the plot of loss and accuracy of each model
    diagram = PyplotDiagram()
    diagram.addAsSeriess(
        {"Original 2015 Model": model_2015_result, "Original 2020 Model": model_2020_result}
    ).drawSeries().setTitle("Datasets Trained Result")
    PyplotDiagram.showAllPlot()

    # This function do the clean-up and finishing job
    cleanEnv()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        cleanEnv()
