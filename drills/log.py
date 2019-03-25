import datetime
def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)