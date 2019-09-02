import errno
import os
import re
from datetime import datetime
from glob import glob

import numpy as np

from Sample import weathersingle


class Weather(object):
    # This class analyses the weather data to give a quality check for the accelerometers data sets

    def __init__(self, **kwargs):
        self.weatherdatafolder = os.environ[
                                     "MST-STR-MON-WEATHERDATA"] + '/'  # folder in which it is supposed to be stored the weather data files
        self.searchstring = kwargs.get("files", 'weather*2019')
        self.weatherdatafiles = kwargs.get("files", np.sort(glob(self.weatherdatafolder + self.searchstring + "*.txt")))
        self.numberofdatafiles = np.size(self.weatherdatafiles)
        self.date = [os.path.basename(self.weatherdatafiles[eachdate])[11:-4] for eachdate in
                     range(self.numberofdatafiles)]

        # Seting directories
        self.resultsfolder = os.environ["MST-STR-MON-WEATHERRESULTS"] + '/'
        try:
            os.makedirs(self.resultsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def Selectionfiles(self, dates, **kwargs):
        self.period = kwargs.get('period', 'daily')
        self.initialdate = datetime(int(dates[0][:4]), int(dates[0][5:7]), int(dates[0][8:10]))
        self.finaldate = datetime(int(dates[1][:4]), int(dates[1][5:7]), int(dates[1][8:10]))
        self.sizeoffiles = np.size(self.weatherdatafiles)
        self.datestring = np.zeros((self.sizeoffiles)).astype(str)
        self.localdates = np.zeros((self.sizeoffiles)).astype(datetime)

        for eachdate in range(self.sizeoffiles):
            self.redate = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(self.weatherdatafiles[eachdate]))
            self.datestring[eachdate] = str(self.redate.group())
            self.localdates[eachdate] = datetime(int(self.datestring[eachdate][0:4]),
                                                 int(self.datestring[eachdate][4:6]),
                                                 int(self.datestring[eachdate][6:8]))
        self.selectedindexes = np.where((self.localdates >= self.initialdate) & (self.localdates <= self.finaldate))

        # Considering that data is taken everyday
        if self.period == 'daily':
            self.selecteddatafiles = self.weatherdatafiles[self.selectedindexes]
        elif self.period == 'weekly':
            self.selecteddatafiles = self.weatherdatafiles[self.selectedindexes][::7]
        elif self.period == 'montly':
            self.selecteddatafiles = self.weatherdatafiles[self.selectedindexes][::30]
        elif self.period == 'yearly':
            self.selecteddatafiles = self.weatherdatafiles[self.selectedindexes][::365]
        return self.selecteddatafiles

    def Analysis(self, **kwargs):
        self.timeofacquisition = kwargs.get('timeofacquisition', (0, 24))
        self.windmeanthreshold = kwargs.get('windmeanthreshold', 1)
        self.winddirminvariance = kwargs.get('winddirminvariance', 180)
        self.files = kwargs.get('files', self.weatherdatafiles)
        self.numberoselectedfdatafiles = np.size(self.files)
        self.numofvariables = 8
        self.Flags = np.zeros((self.numberofdatafiles, 2))
        self.meanvalues, self.variance, self.minvalues, self.maxvalues = [
            np.zeros((self.numberofdatafiles, self.numofvariables)) for i in range(4)]

        for filenum in range(self.numberoselectedfdatafiles):
            print("Generating graphs for Weather:")
            print(self.files[filenum])
            weatherclass = weathersingle.Weather(os.path.basename(self.files[filenum]))
            DONTRUNFLAG = weatherclass.Analysis(timeofacquisition=self.timeofacquisition)
            if DONTRUNFLAG == True:
                print('Weather files already generated for this date')
            else:
                self.meanvalues[filenum], self.variance[filenum], self.minvalues[filenum], self.maxvalues[
                    filenum] = weatherclass.Statistics()
                self.Flags[filenum] = weatherclass.Qualitycheck(windmeanthreshold=self.windmeanthreshold,
                                                                winddirminvariance=self.winddirminvariance)
        return self.Flags

    def Selectioncriteria(self):
        self.goodfileindexes = np.where((self.Flags[:, 0] == 1) & (self.Flags[:, 1] == 1))
        self.goodfileindexes = self.goodfileindexes[0]
        self.goodfiles = self.weatherdatafiles[self.goodfileindexes]
        return self.goodfiles


if __name__ == "__main__":
    C = Weather()
    today = str(datetime.today())[:10]
    Selectedfiles = C.Selectionfiles(dates=['2019-08-16', today])
    Flags = C.Analysis(files=Selectedfiles, timeofacquisition=(6, 7))
    Goodfiles = C.Selectioncriteria()
