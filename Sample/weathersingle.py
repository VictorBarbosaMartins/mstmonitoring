import errno
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import chisquare

import definitions as names


class Weather(object):
    """
    This class analyses the weather data to give a quality check for the accelerometers data set
    :param weatherfilename: name of the weather data file (with path)
    :type weatherfilename: str
    """

    def __init__(self, weatherfilename):
        self.weatherdatafile = weatherfilename
        self.weatherdatafolder = os.environ[
                                     "MST-STR-MON-WEATHERDATA"] + '/'  # folder in which it is supposed to be stored
        # the weather data files

        # Seting directories
        self.resultsfolder = os.environ["MST-STR-MON-WEATHERRESULTS"] + '/'
        try:
            os.makedirs(self.resultsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def Analysis(self, **kwargs):
        """Executes the analysis of the weather data
        It plots all the variables for the specified period. Returns a Flag which tells us
        if the data was already analysed (in this case no further analysis takes place)

        :param kwargs: The keyword timeofacquisition is used for constraining the time window in which the data is
        analysed
        :type kwargs: tuple of int
        :return: Flag which indicates if data was already analysed (1 = yes)
        :rtype: int

        """

        self.timeofacquisition = kwargs.get('timeofacquisition', (0, 24))
        self.weatherinfo = np.loadtxt(self.weatherdatafolder + self.weatherdatafile, delimiter='\t', skiprows=1,
                                      dtype=float)
        self.sizeoffile = np.size(self.weatherinfo, axis=0)
        variablenames = ['time', 'outsidetemperature', 'insidetemperature', 'outsidehumidity', 'insidehumidity',
                         'pressure',
                         'windspeed', 'winddirection', 'avgwindspeed', 'rainrate']
        self.dictofvariables = {}
        monthdict = {1: "Jan.", 2: "Feb.", 3: "Mar.", 4: "Apr.", 5: "May", 6: "Jun.", 7: "Jul.", 8: "Aug.", 9: "Sep.",
                     10: "Okt.", 11: "Nov.", 12: "Dez"}
        self.month = monthdict[int(self.weatherdatafile[15:17])]
        self.day = str(self.weatherdatafile[17:19])
        self.year = str(self.weatherdatafile[11:15])

        # Creating dictionary
        for column in range(np.size(self.weatherinfo, axis=1)):
            self.dictofvariables[variablenames[column]] = self.weatherinfo[:, column]

        # Setting just the time and temperature variables in order to decide if run the analysis or not
        self.dictofvariables['time'] = 24 * (self.dictofvariables['time'] - self.dictofvariables['time'][0]) / (
                self.dictofvariables['time'][-1] - self.dictofvariables['time'][0])
        self.condition = (self.dictofvariables['time'] > self.timeofacquisition[0]) * (
                self.dictofvariables['time'] < self.timeofacquisition[1]) == 1
        self.time = self.dictofvariables['time'][self.condition]
        self.outsidetemperature = self.dictofvariables['outsidetemperature'][self.condition]
        self.insidetemperature = self.dictofvariables['insidetemperature'][self.condition]

        nonzeroouttemp = self.outsidetemperature != 0
        self.outsidetemperature = self.outsidetemperature[nonzeroouttemp]
        nonzerointemp = self.insidetemperature != 0
        self.insidetemperature = self.dictofvariables['insidetemperature'][self.condition]
        self.insidetemperature = self.insidetemperature[nonzerointemp]
        OUTTEMP = self.resultsfolder + self.year + self.month + self.day + names.TEMPERATURE + '.png'
        if os.path.isfile(OUTTEMP):
            DONTRUNFLAG = 1
            print("File " + OUTTEMP + " already exists.")
            print("If you wish to run analysis, please delete results")

        else:
            DONTRUNFLAG = 0

            # Temperature
            figtemperature = plt.figure(figsize=(8, 6))
            xytemperature = figtemperature.add_subplot(111)

            xytemperature.plot(self.time[nonzeroouttemp], self.outsidetemperature,
                               label='outside temperature', linewidth=0.5)
            xytemperature.plot(self.time[nonzerointemp], self.insidetemperature,
                               label='inside temperature', linewidth=0.5)
            xytemperature.set_xlim(self.timeofacquisition[0], self.timeofacquisition[1])
            xytemperature.set_xlabel('Time (h)', fontsize=15)
            xytemperature.set_ylabel('Temperature (°C)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            xytemperature.set_title(self.month + ', ' + self.day + ', ' + self.year, fontsize=20)
            plt.legend(fontsize=15)
            figtemperature.savefig(OUTTEMP)
            plt.close()

            # Humidity
            fighumidity = plt.figure(figsize=(8, 6))
            xyhumidity = fighumidity.add_subplot(111)
            self.outhumidity = self.dictofvariables['outsidehumidity'][self.condition]
            self.inhumidity = self.dictofvariables['insidehumidity'][self.condition]
            nonzeroouthum = self.outhumidity != 0
            nonzeroinhum = self.inhumidity != 0
            self.outhumidity = self.outhumidity[nonzeroouthum]
            self.inhumidity = self.inhumidity[nonzeroinhum]

            xyhumidity.plot(self.time[nonzeroouthum], self.outhumidity, label='outside humidity')
            xyhumidity.plot(self.time[nonzeroinhum], self.inhumidity, label='inside humidity')
            xyhumidity.set_xlim(self.timeofacquisition[0], self.timeofacquisition[1])
            xyhumidity.set_xlabel('Time (h)', fontsize=20)
            xyhumidity.set_ylabel('Relative humidity (%)', fontsize=20)
            xyhumidity.set_title(self.month + ', ' + self.day + ', ' + self.year, fontsize=20)
            plt.legend(fontsize=15)
            fighumidity.savefig(self.resultsfolder + self.year + self.month + self.day + names.HUMIDITY + '.png')
            plt.close()

            # Pressure
            figpressure = plt.figure(figsize=(8, 6))
            xypressure = figpressure.add_subplot(111)
            self.pressure = self.dictofvariables['pressure'][self.condition]
            nonzeropressure = self.pressure != 0
            xypressure.scatter(self.time[nonzeropressure], self.pressure[nonzeropressure], label='Pressure',
                               linewidth=1)
            xypressure.set_xlim(self.timeofacquisition[0], self.timeofacquisition[1])
            xypressure.set_xlabel('Time (h)', fontsize=15)
            xypressure.set_ylabel('Pressure (mm Hg)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            xypressure.set_title(self.month + ', ' + self.day + ', ' + self.year, fontsize=15)
            plt.legend(fontsize=15)
            figpressure.savefig(self.resultsfolder + self.year + self.month + self.day + names.PRESSURE + '.png')
            plt.close()

            # Wind
            figwind = plt.figure(figsize=(8, 8))
            xywind = figwind.add_subplot(111)
            self.windspeed = self.dictofvariables['windspeed'][self.condition]
            maxspeed = np.amax(self.windspeed)
            self.winddirection = self.dictofvariables['winddirection'][self.condition]

            for step in range(np.size(self.winddirection)):
                # Defining direction of the wind
                '''if (self.winddirection[step]>0) & (self.winddirection[step]<90):
                    xdirec = np.cos(np.pi / 360 * self.winddirection[step])
                    ydirec = np.sin(np.pi / 360 * self.winddirection[step])
                elif (self.winddirection[step]) > 90 & (self.winddirection[step] < 180):
                    xdirec = np.cos(np.pi / 360 * self.winddirection[step])
                    ydirec = np.sin(np.pi / 360 * self.winddirection[step])
                elif (self.winddirection[step] > 180) & (self.winddirection[step] < 270):
                    xdirec = np.cos(np.pi / 360 * self.winddirection[step])
                    ydirec = np.sin(np.pi / 360 * self.winddirection[step])
                elif (self.winddirection[step] > 270) & (self.winddirection[step] < 360):'''
                xdirec = np.cos(np.pi / 180 * self.winddirection[step])
                ydirec = np.sin(np.pi / 180 * self.winddirection[step])

                # Drawing lines in the diagram
                windarrowx = [0, self.windspeed[step] * xdirec]
                windarrowy = [0, self.windspeed[step] * ydirec]
                linewind = mlines.Line2D(windarrowx, windarrowy, c='red')
                xywind.add_line(linewind)
            circle = plt.Circle((0, 0), maxspeed, color='black', fill=False, linestyle='--')
            plt.text(np.around(maxspeed, 1) - .45, 0, str(np.around(maxspeed, 1)))
            onethirdcircle = plt.Circle((0, 0), 1 / 3 * maxspeed, color='black', fill=False, linestyle='--')
            plt.text(np.around(1 / 3 * maxspeed, 1) - .45, 0, str(np.around(1 / 3 * maxspeed, 1)))
            twothirdscircle = plt.Circle((0, 0), 2 / 3 * maxspeed, color='black', fill=False, linestyle='--')
            plt.text(np.around(2 / 3 * maxspeed, 1) - .45, 0, str(np.around(2 / 3 * maxspeed, 1)))
            xywind.add_artist(circle)
            xywind.add_artist(onethirdcircle)
            xywind.add_artist(twothirdscircle)
            xywind.set_title('Wind diagram: ' + self.month + ', ' + self.day + ', ' + self.year + ', ' + str(
                self.timeofacquisition[0]) + 'h - ' + str(self.timeofacquisition[1]) + 'h', fontsize=15)
            xywind.set_xlim(-maxspeed, maxspeed)
            xywind.set_ylim(-maxspeed, maxspeed)
            xywind.set_xlabel('Wind speed X-direction (m/s)', fontsize=15)
            xywind.set_ylabel('Wind speed Y-direction (m/s)', fontsize=15)
            figwind.savefig(self.resultsfolder + self.year + self.month + self.day + names.WIND + '.png')
            plt.close()

            # Rain rate
            figrainrate = plt.figure(figsize=(8, 8))
            xyrainrate = figrainrate.add_subplot(111)
            self.rainrate = self.dictofvariables['rainrate'][self.condition]
            nonzeropressure = self.pressure != 0
            xyrainrate.scatter(self.time, self.rainrate, label='Rain rate', linewidth=1)
            xyrainrate.set_xlim(self.timeofacquisition[0], self.timeofacquisition[1])
            xyrainrate.set_xlabel('Time (h)', fontsize=15)
            xyrainrate.set_ylabel('Rain rate', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            xyrainrate.set_title(self.month + ', ' + self.day + ', ' + self.year, fontsize=15)
            plt.legend(fontsize=15)
            figrainrate.savefig(self.resultsfolder + self.year + self.month + self.day + names.RAINRATE + '.png')
            # plt.show()
            plt.close()
        return DONTRUNFLAG

    def Statistics(self, DONTRUNFLAG: bool) -> tuple[float, float, float, float]:
        """Derive the statistical information about the data set. The quality check is created based on this
        statistics. Returns mean, variance, min and max values for all the variables

        :param DONTRUNFLAG: Flag not to run analyses again.

        :return meanvalues: Mean values for every weather variable,
        :return variance: Variance for every weather variable,
        :return minvalues: Min values for every weather variable,
        :return maxvalues: Max values for every weather variable,
        """
        if DONTRUNFLAG == 1:
            print("If you wish to run analysis, please delete results")

        else:

            # Statistics
            self.variables = [self.outsidetemperature, self.insidetemperature, self.outhumidity, self.inhumidity,
                              self.pressure, self.windspeed, self.winddirection, self.rainrate]
            self.numofvariables = np.size(self.variables, axis=0)
            dicttreateddata = {v: self.variables[v] for v in range(self.numofvariables)}
            self.meanvalues, self.variance, self.minvalues, self.maxvalues = [np.zeros(self.numofvariables) for i in
                                                                              range(4)]
            for variablenum in range(self.numofvariables):
                describedstats = scipy.stats.describe(dicttreateddata[variablenum])
                self.meanvalues[variablenum] = describedstats.mean
                self.variance[variablenum] = describedstats.variance
                self.minvalues[variablenum] = describedstats.minmax[0]
                self.maxvalues[variablenum] = describedstats.minmax[1]
                print(describedstats)
            return self.meanvalues, self.variance, self.minvalues, self.maxvalues

    def Qualitycheck(self, DONTRUNFLAG: int, **kwargs) -> tuple:
        """
        Compares the results from the statistics of a dataset to the pre-defined thresholds and returns the Flags
        for wind speed and wind direction respectively. If a Flag is 1, the dataset is good for further analysis
        Parameters
        ----------
        :param DONTRUNFLAG: Flag indicating if the code was already ran before
        :type DONTRUNFLAG: int
        :param kwargs: winddirminvariance,
        :type kwargs:
        :return WindSpeedFlag: Flag confirming if the threshold of the average wind speed was reached
        :rtype WindSpeedFlag: int
        :return WindDirectionFlag: Flag confirming if the std threshold direction  was reached
        :rtype WindDirectionFlag: int
        """

        self.windmeanthreshold = kwargs.get('windthreshold', 1.1)
        self.winddirminvariance = kwargs.get('winddirminvariance', 180)

        if DONTRUNFLAG == 1:
            print("If you wish to run analysis, please delete results")

        else:

            WindSpeedFlag, WindDirectionFlag = 0, 0
            if self.meanvalues[5] >= self.windmeanthreshold:
                WindSpeedFlag = 1
            else:
                WindSpeedFlag = 0
                print('Wind is not strong enough for data taking on ' + str(self.month) + ', ' + str(
                    self.day) + ', ' + str(self.year) + ' from ' + str(self.timeofacquisition[0]) + 'h to ' + str(
                    self.timeofacquisition[1]) + 'h')
                print('This dataset will be excluded from analysis!')

            if self.variance[6] >= self.winddirminvariance:
                WindDirectionFlag = 1
            else:
                WindDirectionFlag = 0
                print('Wind direction did not vary much for data taking on ' + str(self.month) + ', ' + str(
                    self.day) + ', ' + str(self.year) + ' from ' + str(self.timeofacquisition[0]) + 'h to ' + str(
                    self.timeofacquisition[1]) + 'h')
                print('This dataset will be excluded from analysis!')

            return WindSpeedFlag, WindDirectionFlag


if __name__ == "__main__":
    A = Weather('weatherData20190703.txt')
    DONTRUNFLAG = A.Analysis(timeofacquisition=(7, 7.5))
    help(Weather)
    # A.Statistics(DONTRUNFLAG)
    # A.Qualitycheck(DONTRUNFLAG)
