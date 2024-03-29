import errno
import math
import os
import re
from datetime import datetime
from glob import glob

import matplotlib.lines as mlines
import numpy as np
from matplotlib import pyplot as plt

import definitions as names
from Sample import omasingle, weathermultiple


class RunOMA(object):
    '''This class runs the OMA for every file in the project folder'''

    def __init__(self, **kwargs):
        self.goodfiles = kwargs.get('files', 'all')
        self.sensors = kwargs.get('sensors', np.array([12, 13, 23, 10, 11, 0, 4, 5, 6]))
        self.desiredmaxfreq = kwargs.get('desiredmaxfreq', 10)
        self.resolution = kwargs.get('resolution', 2048 * 2)
        self.numoflinestoplot = kwargs.get('numoflinestoplot', 3)
        # self.homefolder = os.path.normpath(os.getcwd() + os.sep + os.pardir) #home directory of the system
        self.homefolder = os.environ["MST-STR-MON-HOME"] + '/'
        self.datafolder = os.environ["MST-STR-MON-DATA"] + '/'
        # self.searchstring = kwargs.get('searchstring', "structure*1.6.100*201*" + '_15' + names.MERGING + '*')
        self.searchstring = kwargs.get('searchstring', "20*" + '_15' + names.MERGING + '*.txt')
        self.datafilenames = np.sort(glob(
            self.datafolder + self.searchstring))  # folder in which it is supposed to be stored the accelerometers data files
        print('datafilenames', self.datafilenames)
        self.sizeofgoodfiles = np.size(self.datafilenames)
        self.date = np.zeros((self.sizeofgoodfiles)).astype(str)
        for eachdate in range(self.sizeofgoodfiles):
            self.redate = (re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(self.datafilenames[eachdate])))
            # date = datetime.datetime.strptime(self.redate.group(), '%Y-%m-%d').date()
            self.date[eachdate] = str(self.redate.group())
        if type(
                self.goodfiles) == str and self.goodfiles == 'all':  # Using all the files in the expected folder. if not, use the selected datasets
            print("Tracking all files")
        else:
            '''print("Tracking selected files")
            self.numberofgooddatafiles = np.size(self.goodfiles)
            self.gooddate = np.zeros((self.numberofgooddatafiles)).astype(str)
            self.goodindexes = np.zeros((self.numberofgooddatafiles)).astype(int)
            for eachdate in range(self.numberofgooddatafiles):
                self.redate = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(self.goodfiles[eachdate]))
                # date = datetime.datetime.strptime(self.redate.group(), '%Y-%m-%d').date()
                self.gooddate[eachdate] = str(self.redate.group())[:4] + '-' + str(self.redate.group())[
                                                                               4:6] + '-' + str(self.redate.group())[
                                                                                            6:8]
                print(self.date)
                print(self.gooddate[eachdate])
                print(np.where(self.date == self.gooddate[eachdate])[0])
                self.goodindexes[eachdate] = np.where(self.date == self.gooddate[eachdate])[0]

            self.datafilenames = self.datafilenames[self.goodindexes]
            self.sizeofgoodfiles = self.numberofgooddatafiles
            self.date = self.gooddate

        self.basename = os.path.basename(self.datafilenames[0])[:91]'''
            print("Tracking selected files")
            self.sizeofweatherfiles = np.size(self.goodfiles)
            self.gooddate = np.zeros((self.sizeofweatherfiles)).astype(str)
            # self.goodindexes = np.zeros((self.numberofgooddatafiles)).astype(int)
            self.gooddataindexes, istherefilethatday, self.newweatherfilenames, self.newdate = [], [], [], []
            for eachdate in range(self.sizeofweatherfiles):
                self.redate = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(self.goodfiles[eachdate]))

                # date = datetime.datetime.strptime(self.redate.group(), '%Y-%m-%d').date()
                self.gooddate[eachdate] = str(self.redate.group())[:4] + '-' + str(self.redate.group())[
                                                                               4:6] + '-' + str(self.redate.group())[
                                                                                            6:8]
            for eachweatherdata in range(self.sizeofweatherfiles):
                newindex = np.where(self.date == self.gooddate[eachweatherdata])[0]
                self.gooddataindexes.append(newindex)
            self.gooddataindexes = np.array(self.gooddataindexes)
            self.gooddataindexes = np.array(self.gooddataindexes[np.where(self.gooddataindexes >= 0)[0]]).flatten()
            self.gooddataindexes = np.array(self.gooddataindexes, dtype=int)
            print(self.gooddataindexes)
            print(self.datafilenames)
            self.datafilenames = self.datafilenames[self.gooddataindexes]
            # self.datafilenames = self.datafilenames[self.goodindexes[np.where(istherefilethatday==True)[0]]]
            self.sizeofdatafiles = np.size(self.datafilenames)
            self.date = self.gooddate
        self.basename = os.path.basename(self.datafilenames[0])

    def sample(self):
        # for datafilenum in range(self.sizeofweatherfiles):
        # justfilename = os.path.basename(self.datafilenames[datafilenum])
        for datafilenum in range(self.sizeofdatafiles):
            justfilename = os.path.basename(self.datafilenames[datafilenum])
            print('Running OMA for file:', justfilename)
            OMAinscene = omasingle.OMA(justfilename)
            OMAinscene.rawdataplot18()
            OMAinscene.calibrate()
            OMAinscene.sensorshift()
            frequencyrange, singvaluesindecb, left, right, DONTRUNFLAG = OMAinscene.FDD(DONTRUNFLAG=0)
            # OMAinscene.peakpicking()
            if DONTRUNFLAG == 0:
                MACmatrix = OMAinscene.MACfunction()
                OMAinscene.MACplot(MACmatrix)
                newmodalfreq, goodindexinfullrange, newMACmatrix = OMAinscene.MACselection(MACmatrix)
                OMAinscene.MACplot(newMACmatrix)
                OMAinscene.enhancedfdd()


class Track(object):

    # This class analyses the results of the OMA for a series of files, study trends, track modes and detect damage
    def __init__(self, **kwargs):
        # self.MAClimit = kwargs.get("MAClimit",0.95)
        self.MAClimit = kwargs.get("MAClimit", 0.95)
        self.goodfiles = kwargs.get("files", "all")
        self.homefolder = os.environ["MST-STR-MON-HOME"] + '/'  # home directory of the system
        self.datafolder = os.environ["MST-STR-MON-DATA"] + '/'
        self.omaonefilefolder = os.environ["MST-STR-MON-ONEFILEOMA"] + '/'
        # searching specific files: names.MERGING: just merged datas; 100 hz in 2019, for sensors...
        self.searchstring = kwargs.get('searchstring', "20*" + names.MERGING)

        self.EFDD_modalfreq_filenames = np.sort(glob(
            self.omaonefilefolder + self.searchstring + '*' + names.EFDD_FREQ_DAMP + '*.txt'))  # folder in which it is supposed to be stored the accelerometers data files
        self.EFDD_modalshape_filenames = np.sort(glob(
            self.omaonefilefolder + self.searchstring + '*' + names.EFDD_MODE_SHAPE + "*.txt"))  # folder in which it is supposed to be stored the accelerometers data files
        self.shifts_filenames = np.sort(glob(self.omaonefilefolder + self.searchstring + '*' + names.SHIFT + '*.txt'))
        self.shiftsavg_filenames = np.sort(
            glob(self.omaonefilefolder + self.searchstring + '*' + names.SHIFT + 'avg*' + '*.txt'))
        self.numberofdatafiles = np.size(self.EFDD_modalfreq_filenames)
        self.date = np.zeros((self.numberofdatafiles)).astype(str)
        for eachdate in range(self.numberofdatafiles):
            self.redate = (re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(self.EFDD_modalfreq_filenames[eachdate])))
            # self.redate = (re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', os.path.basename(self.EFDD_modalfreq_filenames[eachdate])))

            # date = datetime.datetime.strptime(self.redate.group(), '%Y-%m-%d').date()
            self.date[eachdate] = str(self.redate.group())

        if type(
                self.goodfiles) == str and self.goodfiles == 'all':  # Using all the files in the expected folder. if not, use the selected datasets
            print("Tracking all files")
        else:
            '''self.numberofgooddatafiles = np.size(self.goodfiles)
            self.gooddate = np.zeros((self.numberofgooddatafiles)).astype(str)
            self.goodindexes = np.zeros((self.numberofgooddatafiles)).astype(int)
            for eachdate in range(self.numberofgooddatafiles):
                self.redate = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(self.goodfiles[eachdate]))
                print(self.redate)
                self.gooddate[eachdate] = str(self.redate.group())[:4] + '-' + str(self.redate.group())[
                                                                               4:6] + '-' + str(self.redate.group())[
                                                                                            6:8]
                self.goodindexes[eachdate] = np.where(self.date == self.gooddate[eachdate])[0]

            self.EFDD_modalshape_filenames = self.EFDD_modalshape_filenames[self.goodindexes]
            self.EFDD_modalfreq_filenames = self.EFDD_modalfreq_filenames[self.goodindexes]
            self.shifts_filenames = self.shifts_filenames[self.goodindexes]
            self.numberofdatafiles = self.numberofgooddatafiles
            self.date = self.gooddate
        self.basename = os.path.basename(self.EFDD_modalfreq_filenames[0])[:91]
        # for eachdate in range(self.numberofdatafiles):
        #    self.date = str(self.date[eachdate][0]) + '-' + str(self.date[eachdate][1]) + '-' +str(self.date[eachdate][2])
        # print(self.date)'''
            self.numberofgooddatafiles = np.size(self.goodfiles)
            self.gooddate = np.zeros((self.numberofgooddatafiles)).astype(str)
            self.goodindexes = np.zeros((self.numberofgooddatafiles)).astype(int)

            for eachdate in range(self.numberofgooddatafiles):
                self.redate = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(self.goodfiles[eachdate]))
                # date = datetime.datetime.strptime(self.redate.group(), '%Y-%m-%d').date()
                self.gooddate[eachdate] = str(self.redate.group())[:4] + '-' + str(self.redate.group())[
                                                                               4:6] + '-' + str(self.redate.group())[
                                                                                            6:8]

            self.gooddataindexes = []
            for eachdate in range(self.numberofgooddatafiles):
                newindex = np.where(self.date == self.gooddate[eachdate])[0]
                self.gooddataindexes.append(newindex)

                print(self.gooddate[eachdate])
            self.gooddataindexes = np.array(self.gooddataindexes)

            self.gooddataindexes = np.array(self.gooddataindexes[np.where(self.gooddataindexes >= 0)[0]]).flatten()

            self.gooddataindexes = np.array(self.gooddataindexes, dtype=int)
            # self.goodindexes[eachdate] = np.where(self.date == self.gooddate[eachdate])[0]

            self.date = self.date[self.gooddataindexes]
            print(self.date, 'date')
            self.basename = os.path.basename(self.EFDD_modalfreq_filenames[0])[:91]
            self.EFDD_modalshape_filenames = self.EFDD_modalshape_filenames[self.gooddataindexes]
            self.EFDD_modalfreq_filenames = self.EFDD_modalfreq_filenames[self.gooddataindexes]
            self.shifts_filenames = self.shifts_filenames[self.gooddataindexes]
            self.numberofdatafiles = np.size(self.EFDD_modalshape_filenames)
        # Seting directories
        self.resultsfolder = os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] + '/'
        try:
            os.makedirs(self.resultsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def Trendanalysis(self, **kwargs):
        # Depending on the number of sensors available
        self.numofchannelsindatafile = kwargs.get("numofchannels", 9)
        self.samplingratio = kwargs.get('sampling', 100)
        self.nummaxofcorrel = int(math.factorial(
            self.numofchannelsindatafile // 3) / 2)  # sensor 1 - sensor 2; sensor 2 - sensor 3, sensor 1 - sensor 3
        self.trackfreq = np.zeros(self.numberofdatafiles)
        self.trackdamping = np.zeros(self.numberofdatafiles)

        # Creating plots
        # Tracking modal freq
        figmodalfreq = plt.figure(figsize=(15, 8))
        xymodalfreq = figmodalfreq.add_subplot(111)
        # Tracking damping ratio
        figdamping = plt.figure(figsize=(15, 8))
        xydamping = figdamping.add_subplot(111)
        # Correlation between frequency and damping
        figfreqxdamping = plt.figure(figsize=(15, 8))
        freqxdamping = figfreqxdamping.add_subplot(111)
        # Mode shape plot

        self.EFDD_modalfreq, self.EFDD_modalshape = None, None
        self.numofmodesperfile = np.zeros(self.numberofdatafiles, dtype=int)
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'gray', 'brown']
        print('num of data files', self.numberofdatafiles)
        for filenum in range(self.numberofdatafiles):

            # Read and store information about modal frequencies and mode shapes
            newfilefreq = np.loadtxt(self.EFDD_modalfreq_filenames[filenum])
            self.numofmodesperfile[filenum] = np.size(newfilefreq, axis=1)

            if self.EFDD_modalfreq is not None:
                self.EFDD_modalfreq.append(newfilefreq)
            else:
                self.EFDD_modalfreq = [newfilefreq]

            newfilemode = np.loadtxt(self.EFDD_modalshape_filenames[filenum])
            if self.EFDD_modalshape is not None:
                self.EFDD_modalshape.append(newfilemode)
            else:
                self.EFDD_modalshape = [newfilemode]

            freqxdamping.scatter(newfilefreq[0], 100 * newfilefreq[1])
            freqxdamping.set_xlabel('Frequency (Hz)', fontsize=15)
            freqxdamping.set_ylabel('Damping(%)', fontsize=15)
            # freqxdamping.set_title('Frequency x Damping', fontsize=15)

            xymodalfreq.scatter(np.ones(self.numofmodesperfile[filenum]) * filenum, newfilefreq[0], color='C0')
            xymodalfreq.set_ylabel('Frequency(Hz)', fontsize=15)
            # xymodalfreq.set_title('Modal Frequencies along the days', fontsize=15)

            xydamping.scatter(np.ones(self.numofmodesperfile[filenum]) * filenum, 100 * newfilefreq[1], color='C0')
            xydamping.set_ylabel('Damping (%)', fontsize=15)
            # xydamping.set_title('Damping along the days', fontsize=15)
            # xydamping.set_ylim(0,1)

        self.locs = np.arange(0, self.numberofdatafiles, 1)

        """
        # Tracking mode shape
        #for freq in range(self.numofmodesperfile[filenum]):
        zgr = np.zeros((self.numberofdatafiles,self.numofchannelsindatafile,np.amax(self.numofmodesperfile)))
        xgr, ygr = np.meshgrid(np.arange(self.numberofdatafiles),np.arange(self.numofchannelsindatafile))
        for xiter in range(self.numberofdatafiles):
            tranps = np.transpose(self.EFDD_modalshape[xiter])
            for yiter in range(self.numofchannelsindatafile):
                zgr[xiter, yiter, :self.numofmodesperfile[xiter]] = np.abs(tranps[yiter])

        for dimension in range(self.numofchannelsindatafile):
            figmodeshape = plt.figure(figsize=(15, 8))
            xymodeshape = figmodeshape.add_subplot(111, projection='3d')
            xymodeshape.plot_wireframe(xgr, ygr, zgr[:,:,dimension].transpose())
            xymodeshape.set_title('Mode shape in channel ' + str(dimension), fontsize=15)
            xymodeshape.set_xticks(ticks=self.locs[:-1])
            xymodeshape.set_xticklabels(labels=self.date, rotation=90)
            xymodeshape.set_ylabel('Mode', fontsize=15)
            plt.show()"""

        xymodalfreq.set_xticks(ticks=self.locs)
        xydamping.set_xticks(ticks=self.locs)
        xymodalfreq.set_xticklabels(labels=self.date, rotation=90)
        xydamping.set_xticklabels(labels=self.date, rotation=90)

        # Calculate MAC value between modal frequencies for consecutive days
        self.MAC = np.zeros(
            (self.numberofdatafiles - 1, np.amax(self.numofmodesperfile), np.amax(self.numofmodesperfile)))
        for filenum in range(self.numberofdatafiles - 1):
            for peak1 in range(self.numofmodesperfile[filenum]):
                for peak2 in range(self.numofmodesperfile[filenum + 1]):
                    self.MAC[filenum, peak1, peak2] = np.square(
                        np.dot(self.EFDD_modalshape[filenum][peak1], self.EFDD_modalshape[filenum + 1][peak2])) / (
                                                              np.dot(self.EFDD_modalshape[filenum][peak1],
                                                                     self.EFDD_modalshape[filenum][peak1]) * np.dot(
                                                          self.EFDD_modalshape[filenum + 1][peak2],
                                                          self.EFDD_modalshape[filenum + 1][peak2]))
                    # print(filenum,peak1,peak2,self.MAC[filenum, peak1, peak2])

        # Find which modal frequencies from subsequent days are correlated with one another
        self.correlatedfrequencies = np.array(np.where(self.MAC > self.MAClimit))

        # Count correlations per day
        self.numofcorrelations = np.size(self.correlatedfrequencies, axis=-1)
        self.correlperday = np.zeros(self.numberofdatafiles - 1, dtype=int)
        for filecounter in range(self.numberofdatafiles - 1):
            self.correlperday[filecounter] = np.size(np.where(self.correlatedfrequencies[0] == filecounter))

        # Defining the change factor (variable to monitor) - now in general form, later monitor each specific mode
        self.freq1, self.freq2 = [np.zeros(self.numofcorrelations, dtype=int) for i in range(2)]
        self.modeshapeshift = np.zeros((self.numofcorrelations, self.numofchannelsindatafile))
        self.freqshift, self.dampingshift = [np.zeros(self.numofcorrelations) for i in range(2)]
        self.changefactorfrequency, self.changefactordamping = [np.zeros((self.numberofdatafiles - 1)) for i in
                                                                range(2)]

        for correlationum in range(self.numofcorrelations):
            self.whichfile = (self.correlatedfrequencies[0][correlationum]).astype(int)
            whichfileback = (self.correlatedfrequencies[0][correlationum - 1]).astype(int)
            # Frequencies of the first and the subsequent day in which there is a correlation
            self.freq1[correlationum] = (self.correlatedfrequencies[1][correlationum]).astype(int)
            self.freq2[correlationum] = (self.correlatedfrequencies[2][correlationum]).astype(int)

            # Shifts in the modal parameters for the subsequent correlated days (%)
            # used abs here because some correlated modes are out of phase (180° although they refer to the same freq)
            # self.modeshapeshift[correlationum] = 100* (
            # np.abs(self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]]) - np.abs(
            # self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]])) / \
            # self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]]
            self.modeshapeshift[correlationum] = (
                    np.abs(self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]]) - np.abs(
                self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]]))
            # self.freqshift[correlationum] = 100 * (
            # self.EFDD_modalfreq[self.whichfile + 1][0][self.freq2[correlationum]] -
            # self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]]) / \
            # self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]]
            self.freqshift[correlationum] = (
                    self.EFDD_modalfreq[self.whichfile + 1][0][self.freq2[correlationum]] -
                    self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]])
            # self.dampingshift[correlationum] = 100 * (
            # self.EFDD_modalfreq[self.whichfile + 1][1][self.freq2[correlationum]] -
            # self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]]) / \
            # self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]]
            self.dampingshift[correlationum] = (
                    self.EFDD_modalfreq[self.whichfile + 1][1][self.freq2[correlationum]] -
                    self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]])
            '''self.modeshapeshift[correlationum] = (
                        np.abs(self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]]) - np.abs(
                    self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]]))
            self.freqshift[correlationum] =  (
                        self.EFDD_modalfreq[self.whichfile + 1][0][self.freq2[correlationum]] -
                        self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]])
            self.dampingshift[correlationum] = (
                        self.EFDD_modalfreq[self.whichfile + 1][1][self.freq2[correlationum]] -
                        self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]])'''
            # Ploting each mode shape correlation for each subsequent date
            figshape = plt.figure(figsize=(15, 8))
            axshape = figshape.add_subplot(111)
            axshape.plot(np.arange(self.numofchannelsindatafile),
                         self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]])
            axshape.plot(np.arange(self.numofchannelsindatafile),
                         self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]])
            # axshape.set_title(
            # 'Correlation between subsequent mode shapes on ' + str(self.date[self.whichfile]) + '-' + self.date[
            # self.whichfile + 1], fontsize=15)
            axshape.set_ylabel('Intensity (a.u.)')
            figshape.savefig(self.resultsfolder + self.basename + self.date[self.whichfile] + '-shape-correl-' + str(
                np.around(self.EFDD_modalfreq[self.whichfile + 1][0][self.freq2[correlationum]], 2)) + 'Hz-' + str(
                np.around(self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]], 2)) + 'Hz-' + '.pdf')
            plt.close()

            # Variables for the plot
            linex = [self.whichfile, self.whichfile + 1]
            lineyfreq = [self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]],
                         self.EFDD_modalfreq[self.whichfile + 1][0][self.freq2[correlationum]]]
            lineydamping = [100 * self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]],
                            100 * self.EFDD_modalfreq[self.whichfile + 1][1][self.freq2[correlationum]]]
            linefreq = mlines.Line2D(linex, lineyfreq, c='black')
            linedamping = mlines.Line2D(linex, lineydamping, c='black')
            # XYlinefreqdamp = mlines.Line2D(lineyfreq, lineydamping, c='black')
            xymodalfreq.add_line(linefreq)
            xydamping.add_line(linedamping)
            # for dimension in range(self.numofchannelsindatafile):
            # lineymodeshapes = [np.abs(self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]][dimension]), np.abs(self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]][dimension])]
            # linemodeshapes = mlines.Line2D(linex, lineymodeshapes, c=colors[dimension])
            # xymodeshape.add_line(linemodeshapes)
            # freqxdamping.add_line(XYlinefreqdamp)

        # Plot number of correlation per file: it may also indicate when there is a huge change and therefore no correlation (must have wind selection turned on)
        figcorrelation = plt.figure(figsize=(15, 8))
        xycorrelation = figcorrelation.add_subplot(111)
        xycorrelation.plot(self.locs[:-1], self.correlperday)
        # xycorrelation.set_title('Number of correlation between consecutive days', fontsize=15)
        xycorrelation.set_xticks(ticks=self.locs[:-1])
        xycorrelation.set_xticklabels(labels=self.date, rotation=90)
        xycorrelation.set_ylabel('Number of correlations', fontsize=15)
        figcorrelation.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
            -1] + names.NUM_CORRELATION + '.pdf')
        plt.close()

        '''# Change factor variables
        for filenum in range(self.numberofdatafiles - 1):  # errado essa soma
            firstindex = np.where(self.correlatedfrequencies[0] == filenum)[0][0]
            self.changefactorfrequency[filenum] = np.sum(
                self.freqshift[firstindex:firstindex + self.correlperday[filenum]]) / self.correlperday[filenum]
            self.changefactordamping[filenum] = np.sum(
                self.dampingshift[firstindex:firstindex + self.correlperday[filenum]]) / self.correlperday[filenum]
            #self.changefactorfrequency[filenum] = np.sum(
                #self.freqshift[firstindex:firstindex + self.correlperday[filenum]])
            #self.changefactordamping[filenum] = np.sum(
                #self.dampingshift[firstindex:firstindex + self.correlperday[filenum]])'''
        # Change factor variables and chisquare
        self.chisquarefreq, self.chisquaredamping, self.chisquaremodeshape = [np.zeros((self.numberofdatafiles - 1)) for
                                                                              i in range(3)]
        for filenum in range(self.numberofdatafiles - 1):  # errado essa soma
            if self.correlperday[filenum] != 0:  # if there is no correlation that day
                correlation_in_roi = 0
                firstindex = np.where(self.correlatedfrequencies[0] == filenum)[0][0]
                self.changefactorfrequency[filenum] = np.sum(
                    self.freqshift[firstindex:firstindex + self.correlperday[filenum]]) / self.correlperday[filenum]
                self.changefactordamping[filenum] = np.sum(
                    self.dampingshift[firstindex:firstindex + self.correlperday[filenum]]) / self.correlperday[filenum]
                squarefreq, squaredamp, squaremodeshape = [], [], []
                firstindex = np.where(self.correlatedfrequencies[0] == filenum)[0][0]
                for mode in range(self.correlperday[filenum]):
                    '''squarefreq.append(np.square(self.freqshift[firstindex + mode] / 100) * np.abs(
                        self.EFDD_modalfreq[filenum][0][self.freq1[firstindex + mode]]))
                    squaredamp.append(np.square(self.dampingshift[firstindex + mode] / 100) * np.abs(
                        self.EFDD_modalfreq[filenum][1][self.freq1[firstindex + mode]]))
                    squaremodeshape.append(np.square(self.modeshapeshift[firstindex + mode] / 100) * np.abs(
                        self.EFDD_modalshape[filenum][self.freq1[firstindex + mode]]))

                self.chisquarefreq[filenum] = np.sum(squarefreq) / self.correlperday[filenum]
                self.chisquaredamping[filenum] = np.sum(squaredamp) / self.correlperday[filenum]
                self.chisquaremodeshape[filenum] = np.sum(squaremodeshape) / (
                        self.correlperday[filenum] * self.numofchannelsindatafile)'''
                    freq_to_monitor = self.EFDD_modalfreq[filenum][0][self.freq1[firstindex + mode]]
                    if freq_to_monitor >= 2 and freq_to_monitor <= 4.2:
                        squarefreq.append(np.square(self.freqshift[firstindex + mode] / 100))
                        squaredamp.append(np.square(self.dampingshift[firstindex + mode] / 100))
                        squaremodeshape.append(np.square(self.modeshapeshift[firstindex + mode] / 100))
                        correlation_in_roi += 1
                if correlation_in_roi != 0:
                    self.chisquarefreq[filenum] = np.sum(squarefreq) / correlation_in_roi
                    self.chisquaredamping[filenum] = np.sum(squaredamp) / correlation_in_roi
                    self.chisquaremodeshape[filenum] = np.sum(squaremodeshape) / correlation_in_roi

            else:  # if there is no correlation
                self.changefactorfrequency[filenum] = None
                self.changefactordamping[filenum] = None
                self.freqshift[filenum] = None
                self.dampingshift[filenum] = None
                self.chisquarefreq[filenum] = None
                self.chisquaredamping[filenum] = None
                self.chisquaremodeshape[filenum] = None

        # Monitoring relative changes in the frequency
        figchangefactorfrequency = plt.figure(figsize=(15, 8))
        xychangefactorfrequency = figchangefactorfrequency.add_subplot(111)
        # Monitor relative changes in the damping ratio
        figchangefactordamping = plt.figure(figsize=(15, 8))
        xychangefactordamping = figchangefactordamping.add_subplot(111)
        # Monitor relative changes in the mode shapes
        figchangemodeshape = plt.figure(figsize=(15, 8))
        xychangemodeshape = figchangemodeshape.add_subplot(111)

        xychangefactorfrequency.plot(self.locs[:-1],
                                     self.changefactorfrequency)  # locs[:-1] = range(self.numofdatafiles - 1)
        xychangefactorfrequency.set_ylabel('Frequency Shift/correlation (Hz)', fontsize=15)
        # xychangefactorfrequency.set_title('Frequency Shift for correlated modes', fontsize=15)
        xychangefactorfrequency.set_xticks(ticks=self.locs[:-1])
        xychangefactorfrequency.set_xticklabels(labels=self.date, rotation=90)
        figchangefactorfrequency.tight_layout()
        xychangefactordamping.plot(range(self.numberofdatafiles - 1), self.changefactordamping)
        xychangefactordamping.set_ylabel('Damping deviation/correlation (%)', fontsize=15)
        # xychangefactordamping.set_xlabel('Correlation', fontsize=15)
        # xychangefactordamping.set_title('Change factor in Damping for correlated modes', fontsize=15)
        xychangefactordamping.set_xticks(ticks=self.locs[:-1])
        xychangefactordamping.set_xticklabels(labels=self.date, rotation=90)
        figchangefactordamping.tight_layout()

        figmodalfreq.tight_layout()
        figdamping.tight_layout()
        if type(self.goodfiles) == str and self.goodfiles == 'all':
            figchangefactorfrequency.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHANGE_FACTOR_FREQ + '.pdf')
            figchangefactordamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHANGE_FACTOR_DAMP + '.pdf')
            figmodalfreq.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.TRACK_FREQ + '.pdf')
            figdamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.TRACK_DAMP + '.pdf')
            figfreqxdamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.FREQXDAMP + '.pdf')
            # figmodeshape.savefig(self.resultsfolder + self.basename +self.date[0] + 'until' + self.date[-1] + names.ALLFILES + names.EFDD_MODE_SHAPE + '.png')

        else:
            figchangefactorfrequency.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHANGE_FACTOR_FREQ + '.pdf')
            figchangefactordamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHANGE_FACTOR_DAMP + '.pdf')
            figmodalfreq.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.TRACK_FREQ + '.pdf')
            figdamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.TRACK_DAMP + '.pdf')
            figfreqxdamping.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.FREQXDAMP + '.pdf')
            # figmodeshape.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[-1] + names.SELECTEDFILES + names.EFDD_MODE_SHAPE + '.png')
        plt.close()

        # Mode shape visualization

        # Chi square indicator
        '''self.chisquarefreq, self.chisquaredamping, self.chisquaremodeshape = [np.zeros((self.numberofdatafiles - 1)) for
                                                                              i in range(3)]
        for filenum in range(self.numberofdatafiles - 1):
            squarefreq, squaredamp, squaremodeshape = [], [], []
            firstindex = np.where(self.correlatedfrequencies[0] == filenum)[0][0]
            for mode in range(self.correlperday[filenum]):
                squarefreq.append(np.square(self.freqshift[firstindex + mode] / 100) * np.abs(
                    self.EFDD_modalfreq[filenum][0][self.freq1[firstindex + mode]]))
                squaredamp.append(np.square(self.dampingshift[firstindex + mode] / 100) * np.abs(
                    self.EFDD_modalfreq[filenum][1][self.freq1[firstindex + mode]]))
                squaremodeshape.append(np.square(self.modeshapeshift[firstindex + mode] / 100) * np.abs(
                    self.EFDD_modalshape[filenum][self.freq1[firstindex + mode]]))

            self.chisquarefreq[filenum] = np.sum(squarefreq) / self.correlperday[filenum]
            self.chisquaredamping[filenum] = np.sum(squaredamp) / self.correlperday[filenum]
            self.chisquaremodeshape[filenum] = np.sum(squaremodeshape) / (
                        self.correlperday[filenum] * self.numofchannelsindatafile)'''

        figchifreq = plt.figure(figsize=(15, 8))
        xychifreq = figchifreq.add_subplot(111)
        xychifreq.plot(self.locs[1:], self.chisquarefreq)
        xychifreq.set_ylabel(r'$\alpha$ (Hz)', fontsize=15)
        # xychifreq.set_title(r'$\chi ^2$ for modal frequencies', fontsize=15)
        xychifreq.set_xticks(ticks=self.locs[:-1])
        xychifreq.set_xticklabels(labels=self.date, rotation=90)

        figchidamp = plt.figure(figsize=(15, 8))
        xychidamp = figchidamp.add_subplot(111)
        xychidamp.plot(self.locs[1:], self.chisquaredamping)
        xychidamp.set_ylabel(r'$\beta$', fontsize=15)
        # xychidamp.set_title(r'$\chi ^2$ for damping ratio', fontsize=15)
        xychidamp.set_xticks(ticks=self.locs[:-1])
        xychidamp.set_xticklabels(labels=self.date, rotation=90)

        figchimodeshape = plt.figure(figsize=(15, 8))
        xychimodeshape = figchimodeshape.add_subplot(111)
        xychimodeshape.plot(self.locs[1:], self.chisquaremodeshape)
        xychimodeshape.set_ylabel(r'$\chi ^2$', fontsize=15)
        xychimodeshape.set_title(r'$\chi ^2$ for mode shape', fontsize=15)
        xychimodeshape.set_xticks(ticks=self.locs[:-1])
        xychimodeshape.set_xticklabels(labels=self.date, rotation=90)

        figchifreq.tight_layout()
        figchidamp.tight_layout()
        figchimodeshape.tight_layout()
        if type(self.goodfiles) == str and self.goodfiles == 'all':
            figchifreq.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_FREQ + '.pdf')
            figchidamp.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_DAMPING + '.pdf')
            figchimodeshape.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_MODE_SHAPE + '.pdf')

        else:
            figchifreq.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_FREQ + '.pdf')
            figchidamp.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_DAMPING + '.pdf')
            figchimodeshape.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_MODE_SHAPE + '.pdf')
        plt.close()

        # ALPHA AND BETA IN COMPARISON TO THE FIRST DAY
        # Calculate MAC value between modal frequencies for consecutive days
        self.MAC_day0 = np.zeros(
            (self.numberofdatafiles - 1, np.amax(self.numofmodesperfile), np.amax(self.numofmodesperfile)))
        for filenum in range(self.numberofdatafiles - 1):
            for peak1 in range(self.numofmodesperfile[0]):
                for peak2 in range(self.numofmodesperfile[filenum + 1]):
                    self.MAC_day0[filenum, peak1, peak2] = np.square(
                        np.dot(self.EFDD_modalshape[0][peak1], self.EFDD_modalshape[filenum + 1][peak2])) / (
                                                                   np.dot(self.EFDD_modalshape[0][peak1],
                                                                          self.EFDD_modalshape[0][peak1]) * np.dot(
                                                               self.EFDD_modalshape[filenum + 1][peak2],
                                                               self.EFDD_modalshape[filenum + 1][peak2]))
                    # print(filenum,peak1,peak2,self.MAC[filenum, peak1, peak2])

        # Find which modal frequencies from subsequent days are correlated with one another
        self.correlatedfrequencies_day0 = np.array(np.where(self.MAC_day0 > self.MAClimit))

        # Count correlations per day
        self.numofcorrelations_day0 = np.size(self.correlatedfrequencies_day0, axis=-1)
        self.correlperday_day0 = np.zeros(self.numberofdatafiles - 1, dtype=int)
        for filecounter in range(self.numberofdatafiles - 1):
            self.correlperday_day0[filecounter] = np.size(np.where(self.correlatedfrequencies_day0[0] == filecounter))

        # Defining the change factor (variable to monitor) - now in general form, later monitor each specific mode
        self.freq1_day0, self.freq2_day0 = [np.zeros(self.numofcorrelations_day0, dtype=int) for i in range(2)]
        self.modeshapeshift_day0 = np.zeros((self.numofcorrelations_day0, self.numofchannelsindatafile))
        self.freqshift_day0, self.dampingshift_day0 = [np.zeros(self.numofcorrelations_day0) for i in range(2)]
        self.changefactorfrequency_day0, self.changefactordamping_day0 = [np.zeros((self.numberofdatafiles - 1)) for i
                                                                          in
                                                                          range(2)]

        for correlationum in range(self.numofcorrelations_day0):
            self.whichfile_day0 = (self.correlatedfrequencies_day0[0][correlationum]).astype(int)
            whichfileback_day0 = (self.correlatedfrequencies_day0[0][correlationum - 1]).astype(int)
            # Frequencies of the first and the subsequent day in which there is a correlation
            self.freq1_day0[correlationum] = (self.correlatedfrequencies_day0[1][correlationum]).astype(int)
            self.freq2_day0[correlationum] = (self.correlatedfrequencies_day0[2][correlationum]).astype(int)
            self.modeshapeshift_day0[correlationum] = (
                    np.abs(self.EFDD_modalshape[self.whichfile_day0 + 1][self.freq2_day0[correlationum]]) - np.abs(
                self.EFDD_modalshape[0][self.freq1_day0[correlationum]]))
            self.freqshift_day0[correlationum] = (
                        self.EFDD_modalfreq[self.whichfile_day0 + 1][0][self.freq2_day0[correlationum]] -
                        self.EFDD_modalfreq[0][0][self.freq1_day0[correlationum]])
            self.dampingshift_day0[correlationum] = (
                        self.EFDD_modalfreq[self.whichfile_day0 + 1][1][self.freq2_day0[correlationum]] -
                        self.EFDD_modalfreq[0][1][self.freq1_day0[correlationum]])

        # Change factor variables and chisquare
        self.chisquarefreq_day0, self.chisquaredamping_day0, self.chisquaremodeshape_day0 = [
            np.zeros((self.numberofdatafiles - 1)) for
            i in range(3)]
        for filenum in range(self.numberofdatafiles - 1):  # errado essa soma
            if self.correlperday_day0[filenum] != 0:  # if there is no correlation that day
                correlation_in_roi = 0
                firstindex = np.where(self.correlatedfrequencies_day0[0] == filenum)[0][0]
                self.changefactorfrequency_day0[filenum] = np.sum(
                    self.freqshift_day0[firstindex:firstindex + self.correlperday_day0[filenum]]) / \
                                                           self.correlperday_day0[filenum]
                self.changefactordamping_day0[filenum] = np.sum(
                    self.dampingshift_day0[firstindex:firstindex + self.correlperday_day0[filenum]]) / \
                                                         self.correlperday_day0[filenum]
                squarefreq, squaredamp, squaremodeshape = [], [], []
                firstindex = np.where(self.correlatedfrequencies_day0[0] == filenum)[0][0]
                for mode in range(self.correlperday_day0[filenum]):
                    freq_to_monitor = self.EFDD_modalfreq[0][0][self.freq1_day0[firstindex + mode]]
                    if freq_to_monitor >= 2 and freq_to_monitor <= 4.2:
                        squarefreq.append(np.square(self.freqshift_day0[firstindex + mode]))
                        squaredamp.append(np.square(self.dampingshift_day0[firstindex + mode] / 100))
                        squaremodeshape.append(np.square(self.modeshapeshift_day0[firstindex + mode]))
                        correlation_in_roi += 1
                if correlation_in_roi != 0:
                    self.chisquarefreq_day0[filenum] = np.sum(squarefreq) / correlation_in_roi
                    self.chisquaredamping_day0[filenum] = np.sum(squaredamp) / correlation_in_roi
                    self.chisquaremodeshape_day0[filenum] = np.sum(squaremodeshape) / correlation_in_roi

            else:  # if there is no correlation
                self.changefactorfrequency_day0[filenum] = None
                self.changefactordamping_day0[filenum] = None
                self.freqshift_day0[filenum] = None
                self.dampingshift_day0[filenum] = None
                self.chisquarefreq_day0[filenum] = None
                self.chisquaredamping_day0[filenum] = None
                self.chisquaremodeshape_day0[filenum] = None
        figchifreq_day0 = plt.figure(figsize=(15, 8))
        xychifreq_day0 = figchifreq_day0.add_subplot(111)
        xychifreq_day0.plot(self.locs[1:], self.chisquarefreq_day0)
        xychifreq_day0.set_ylabel(r'$\alpha$ (Hz$^2$)', fontsize=15)
        # xychifreq.set_title(r'$\chi ^2$ for modal frequencies', fontsize=15)
        xychifreq_day0.set_xticks(ticks=self.locs[1:])
        xychifreq_day0.set_xticklabels(labels=self.date[1:], rotation=90)
        np.savetxt(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
            -1] + names.ALLFILES + names.CHISQUARE_FREQ + '_day0.txt',
                   np.array([self.locs[1:], self.chisquarefreq_day0]))

        figchidamp_day0 = plt.figure(figsize=(15, 8))
        xychidamp_day0 = figchidamp_day0.add_subplot(111)
        xychidamp_day0.plot(self.locs[1:], 1e10 * self.chisquaredamping_day0)
        xychidamp_day0.set_ylabel(r'10$^{-10}\beta $', fontsize=15)
        # xychidamp.set_title(r'$\chi ^2$ for damping ratio', fontsize=15)
        xychidamp_day0.set_xticks(ticks=self.locs[1:])
        xychidamp_day0.set_xticklabels(labels=self.date[1:], rotation=90)
        np.savetxt(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
            -1] + names.ALLFILES + names.CHISQUARE_DAMPING + '_day0.txt',
                   np.array([self.locs[1:], self.chisquaredamping_day0]))

        figchimodeshape_day0 = plt.figure(figsize=(15, 8))
        xychimodeshape_day0 = figchimodeshape.add_subplot(111)
        xychimodeshape_day0.plot(self.locs[1:], self.chisquaremodeshape_day0)
        xychimodeshape_day0.set_ylabel(r'$\chi ^2$', fontsize=15)
        xychimodeshape_day0.set_title(r'$\chi ^2$ for mode shape', fontsize=15)
        xychimodeshape_day0.set_xticks(ticks=self.locs[1:])
        xychimodeshape_day0.set_xticklabels(labels=self.date[1:], rotation=90)
        figchifreq_day0.tight_layout()
        figchidamp_day0.tight_layout()
        figchimodeshape_day0.tight_layout()
        if type(self.goodfiles) == str and self.goodfiles == 'all':
            figchifreq_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_FREQ + '_day0.pdf')
            figchidamp_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_DAMPING + '_day0.pdf')
            figchimodeshape_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.ALLFILES + names.CHISQUARE_MODE_SHAPE + '_day0.pdf')

        else:
            figchifreq_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_FREQ + '_day0.pdf')
            figchidamp_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_DAMPING + '_day0.pdf')
            figchimodeshape_day0.savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[
                -1] + names.SELECTEDFILES + names.CHISQUARE_MODE_SHAPE + '_day0.pdf')
        plt.close('all')
        '''
        # Shifts between sensors (all values)
        figshift1, figshift2, figshift3 = [plt.figure(figsize=(15, 8)) for i in range(3)]
        xyshift1, xyshift2, xyshift3 = figshift1.add_subplot(111), figshift2.add_subplot(111), figshift3.add_subplot(
            111)
        plotdic = {0: xyshift1, 1: xyshift2, 2: xyshift3}
        figdict = {0: figshift1, 1: figshift2, 2: figshift3}
        titles = ['Sensor 1 - Sensor 2', 'Sensor 1 - Sensor 3', 'Sensor 2 - Sensor 3']
        labels = ['X axis', 'Y axis', 'Z axis']
        color = ['blue', 'red', 'green']
        self.shifts_filenames = np.array(self.shifts_filenames)
        newfileshift = []
        numofshiftfiles = np.size(self.shifts_filenames, axis=0)
        sizeofeachfile = np.zeros((numofshiftfiles))
        for filenum in range(0, numofshiftfiles):

            for relation in range(self.nummaxofcorrel):
                newtext = np.loadtxt(self.shifts_filenames[filenum // self.nummaxofcorrel + relation])
                sizeofeachfile[filenum] = np.size(newtext, axis=-1)
                time = np.linspace(filenum // self.nummaxofcorrel * sizeofeachfile[filenum] / self.samplingratio,
                                   (filenum + 1) // self.nummaxofcorrel * sizeofeachfile[filenum] / self.samplingratio,
                                   sizeofeachfile[filenum])
                for dimension in range(3):
                    plotdic[relation].plot(time[1:], np.power(10, 6) * newtext[dimension, 1:], color=color[dimension],
                                           label=labels[dimension])
                plotdic[relation].set_title('Shift between: ' + titles[relation], fontsize=15)
                plotdic[relation].set_xticklabels(labels=self.date, rotation=90)
                plotdic[relation].set_xticks(ticks=self.locs * sizeofeachfile[filenum] / self.samplingratio)
                plotdic[relation].set_ylabel(r'Shift ($\mu$m)', fontsize=15)
        for correlation in range(self.nummaxofcorrel):
            plotdic[correlation].grid()
            plotdic[correlation].legend(labels)
            figdict[correlation].savefig(
                self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[-1] + names.SHIFT + titles[
                    correlation] + '.png')
        plt.close()
        '''
        """
        # Shifts between sensors (day average)
        self.avgshiftfilenames = np.sort(glob(self.omaonefilefolder + self.searchstring + '*' + names.SHIFTAVG + '*.txt'))
        print(self.avgshiftfilenames)
        numofshiftfiles = np.size(self.avgshiftfilenames)
        newfileshiftavg = np.zeros((numofshiftfiles,self.nummaxofcorrel*3))
        for filenum in range(numofshiftfiles):
            newfileshiftavg[filenum] = np.loadtxt(self.avgshiftfilenames[filenum])
        plt.close()
        figshift1,figshift2,figshift3 = [plt.figure(figsize=(15, 8)) for i in range(3)]
        figdict = {0: figshift1, 1: figshift2, 2: figshift3}
        xyshift1,xyshift2,xyshift3 = figshift1.add_subplot(111), figshift2.add_subplot(111), figshift3.add_subplot(111)
        plotdic = {0: xyshift1, 1: xyshift2, 2: xyshift3}
        for relation in range(0,self.nummaxofcorrel):
            xyshift1.plot(self.locs,np.power(10,6)*newfileshiftavg[:,relation], label=labels[relation], color=color[relation])
            xyshift2.plot(self.locs,np.power(10,6)*newfileshiftavg[:,3*(relation)+1], label=labels[relation], color=color[relation])
            xyshift3.plot(self.locs,np.power(10,6)*newfileshiftavg[:,(relation)*3+2], label=labels[relation], color=color[relation])
            plotdic[relation].set_title('Average Shift between: ' + titles[relation], fontsize=15)
            plotdic[relation].set_xticklabels(labels=self.date[:-1], rotation=90)
            plotdic[relation].set_xticks(ticks=self.locs[:-1])
            plotdic[relation].set_ylabel(r'Shift ($\mu$m)', fontsize=15)
            plotdic[relation].grid()
        for relation in range(0, self.nummaxofcorrel):
            plotdic[relation].legend(labels)
            figdict[relation].savefig(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[-1] + names.SHIFTAVG + titles[relation] + '.png')

        plt.close()
        """
        np.savetxt(self.resultsfolder + 'freq-damping-shift.txt', np.array([self.freqshift, self.dampingshift]))
        np.savetxt(self.resultsfolder + 'modeshape-shift.txt', self.modeshapeshift)
        np.savetxt(self.resultsfolder + 'chisquare.txt',
                   np.array([self.chisquarefreq, self.chisquaremodeshape, self.chisquaredamping]))
        np.savetxt(self.resultsfolder + 'numofcorrelation.txt', self.correlperday)
        np.savetxt(self.resultsfolder + self.basename + self.date[0] + 'until' + self.date[-1] + '_dates.txt',
                   self.date, fmt='%s')
        # np.savetxt(self.resultsfolder + 'shifts.txt', newfileshiftavg)


if __name__ == "__main__":
    # OMA ANALYSIS
    C = weathermultiple.Weather()
    today = str(datetime.today())[:10]
    Selectedfiles = C.Selectionfiles(dates=['2019-08-16', today])
    Flags = C.Analysis(files=Selectedfiles, timeofacquisition=(6, 7))
    Goodfiles = C.Selectioncriteria()
    A = RunOMA(files=Selectedfiles)
    A.sample()

    # TRACKING
    B = Track(files=Selectedfiles)
    B.Trendanalysis(numofchannels=9)