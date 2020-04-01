


def testSimpleML(DataHandler):
    handler = DataHandler(readAllDataNow = False) 
    postData = handler.loadPostDataOfSubjectVideo(1, 'infinity')
    print(postData.shape)

def main(DataHandler):
   testSimpleML(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError