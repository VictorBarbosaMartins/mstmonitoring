import errno
import os
import re
from datetime import datetime
from glob import glob
from matplotlib import pyplot as plt
import numpy as np

from Sample import weathersingle
import definitions as names


class Weather(object):
    '''This class analyses the weather data to give a quality check for the accelerometers data sets'''

    def __init__(self, **kwargs):
        self.weatherdatafolder = os.environ[
                                     "MST-STR-MON-WEATHERDATA"] + '/'  # folder in which it is supposed to be stored the weather data files
        self.searchstring = kwargs.get("files", 'weather*')
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
        '''Select the weather files based on the period'''
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

        self.sizeofnewdate = np.size(self.selecteddatafiles)
        self.newdate = [os.path.basename(self.selecteddatafiles[eachdate])[11:-4] for eachdate in
                        range(self.sizeofnewdate)]
        print(self.newdate)
        return self.selecteddatafiles

    def Analysis(self, **kwargs):
        '''Analysis of the weather data for the selected period'''
        self.timeofacquisition = kwargs.get('timeofacquisition', (0, 24))
        self.windmeanthreshold = kwargs.get('windmeanthreshold', 1)
        self.windmeanmax = kwargs.get('windmeanmax', 2.5)
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
            # DONTRUNFLAG = 0
            try:
                # DONTRUNFLAG = weatherclass.Analysis(timeofacquisition=self.timeofacquisition)
                weatherclass.Analysis(timeofacquisition=self.timeofacquisition)

                # I commented the following because I want to use the selection criteria
                # if DONTRUNFLAG == 1:
                # print('Weather files already generated for this date')
                # elif DONTRUNFLAG == 0:

                # self.meanvalues[filenum], self.variance[filenum], self.minvalues[filenum], self.maxvalues[filenum] = weatherclass.Statistics(DONTRUNFLAG)
                # self.Flags[filenum] = weatherclass.Qualitycheck(DONTRUNFLAG,windmeanthreshold=self.windmeanthreshold,winddirminvariance=self.winddirminvariance, windmeanmax=self.windmeanmax)
                self.meanvalues[filenum], self.variance[filenum], self.minvalues[filenum], self.maxvalues[
                    filenum] = weatherclass.Statistics()
                self.Flags[filenum] = weatherclass.Qualitycheck(windmeanthreshold=self.windmeanthreshold,
                                                                winddirminvariance=self.winddirminvariance,
                                                                windmeanmax=self.windmeanmax)
            except:
                # DONTRUNFLAG = 2
                print("Analysis of the weather data not possible. There might a problem with the data")

        return self.Flags

    def Selectioncriteria(self):
        '''Applies the selection criteria to the analysed weather data'''
        self.goodfileindexes = np.where((self.Flags[:, 0] == 1) & (self.Flags[:, 1] == 1))
        self.goodfileindexes = self.goodfileindexes[0]
        # if quality criteria was applied
        if (np.size(self.selecteddatafiles) != 0):
            self.goodfiles = self.selecteddatafiles[self.goodfileindexes]
        else:
            self.goodfiles = self.weatherdatafiles[self.goodfileindexes]
        return self.goodfiles

    def Windeveryday(self):
        '''Plots the wind for every day'''
        windmean_filenames = np.array(
            [self.resultsfolder + self.newdate[i] + '-meanvalues.txt' for i in range(self.sizeofnewdate)])
        # print('windmean_filenames',windmean_filenames)
        # print('self.newdate',self.newdate)
        meanwind = []
        for file in windmean_filenames:
            try:
                meanvalues = np.loadtxt(file)
                meanwind.append(meanvalues[5])
            except:
                print("There was a problem to open the file" + str(file))
                meanwind.append(np.nan)
        meanwind = np.array(meanwind)

        figwindeveryday = plt.figure(figsize=(15, 8))
        xywindeveryday = figwindeveryday.add_subplot(111)
        xywindeveryday.plot(self.newdate, meanwind)
        xywindeveryday.set_title('Average wind speed throughout the days', fontsize=15)
        # xywindeveryday.set_xticks(ticks=self.locs[:-1])
        xywindeveryday.set_xticklabels(labels=self.newdate, rotation=90)
        xywindeveryday.set_ylabel('Average wind speed (m/s)', fontsize=15)
        figwindeveryday.savefig(self.resultsfolder + self.newdate[0] + 'until' + self.newdate[
            -1] + names.WIND_DAYS + '.png')
        plt.close()

    def Tempeveryday(self):
        '''Plots the temperature for every day'''
        tempmean_filenames = np.array(
            [self.resultsfolder + self.newdate[i] + '-meanvalues.txt' for i in range(self.sizeofnewdate)])
        # print('windmean_filenames',windmean_filenames)
        # print('self.newdate',self.newdate)
        meantemp = []
        for file in tempmean_filenames:
            try:
                meanvalues = np.loadtxt(file)
                meantemp.append(meanvalues[0])  # outside temp
            except:
                print("There was a problem to open the file" + str(file))
                meantemp.append(np.nan)
        meantemp = np.array(meantemp)

        figtempeveryday = plt.figure(figsize=(15, 8))
        xytempeveryday = figtempeveryday.add_subplot(111)
        xytempeveryday.plot(self.newdate, meantemp)
        xytempeveryday.set_title('Average temperature throughout the days', fontsize=15)
        # xywindeveryday.set_xticks(ticks=self.locs[:-1])
        xytempeveryday.set_xticklabels(labels=self.newdate, rotation=90)
        xytempeveryday.set_ylabel('Average temperature (°C)', fontsize=15)
        figtempeveryday.savefig(self.resultsfolder + self.newdate[0] + 'until' + self.newdate[
            -1] + names.TEMP_DAYS + '.png')
        plt.close()


if __name__ == "__main__":
    C = Weather()
    today = str(datetime.today())[:10]
    Selectedfiles = C.Selectionfiles(dates=['2019-08-16', today])
    Flags = C.Analysis(files=Selectedfiles, timeofacquisition=(6, 7))
    Goodfiles = C.Selectioncriteria()