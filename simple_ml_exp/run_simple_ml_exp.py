from matplotlib import pyplot as plt

def testSimpleML(DataHandler):
    handler = DataHandler(readAllDataNow = False) 
    data, postData = handler.getHeadGazeToPointingData(1, 'vertical')
    print(data.shape)
    print(postData.shape)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(postData[:, 0], postData[:, 1])
    plt.show()

def main(DataHandler):
   testSimpleML(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError