from DataHandler import DataHandler

def main():
    handler = DataHandler(readAllDataNow = False)
    #trails = handler.readAllTrails()
    trail = handler.readTrail('random1.csv')
    print()
    print(trail['meta']['frameCount'])
    print(trail['data'].shape)
    print(trail['data'])
    print(trail['data'][-1])

if __name__ == '__main__':
    main()
