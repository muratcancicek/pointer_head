import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from data_handler.run_data_handler import main as testDataHandler
from data_handler.DataHandler import DataHandler
from simple_ml_exp.run_simple_ml_exp import main as testSimpleML

def main():
    testDataHandler()
  #  testSimpleML(DataHandler)

if __name__ == '__main__':
    main()