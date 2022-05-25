from util.PyplotDiagram import PyplotDiagram
from util.console import console
from util.helper import prepareEnv, cleanEnv
from train.single import *
from train.single import TrainedModel
from train.pre_processing import *

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

    # Select the most influential features (first 90%) to the target value
    console.info('Data Cleaning: 1. Feature Selection (30-40s)')
    # Uses the mutual_info_classif method provided by Keras, which is encapsulated in the selectFeatures method (self-implemented)
    features_selected = selectFeatures(df_2015, labelColName="HeartDisease", threshold=0.9)
    print("{} features selected".format(len(features_selected)))
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
    # Check Project_Interactive_Demo.ipynb for more information about this configureation 
    # (Train a model for the 2015 dataset/Test Trainning: Find the best configuration)
    model_2015 = getModelByXYColumn(
        y_train, x_train,
        use_CPU=True, layers=layers,
        fit_epoch=4, compile_loss_function='binary_crossentropy', fit_callbacks=[]
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
    exit(0)

    ### Train a model for the 2015 dataset ############################################################################

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

    # Draw the plot of loss and accuracy of each model
    diagram = PyplotDiagram()
    diagram.drawSeries(
        {"Original 2015 Model": result_2015, "Original 2020 Model": model_2020_result}
    ).setTitle("Datasets Trained Result")
    PyplotDiagram.showAllPlot()

    # This function do the clean-up and finishing job
    cleanEnv()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.err("You have stopped the program manually.")
        cleanEnv()
