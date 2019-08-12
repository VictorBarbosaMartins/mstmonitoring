import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Sample import onefileanalysis
import sys, errno

class RunOMA(object):
    '''This class runs the OMA for every file in the project folder'''
    def __init__(self, **kwargs):
        self.goodfiles = kwargs.get('files', 'all')
        #self.homefolder = os.path.normpath(os.getcwd() + os.sep + os.pardir) #home directory of the system
        self.homefolder = os.path.normpath(os.getcwd()) #home directory of the system

        if type(self.goodfiles) == str and self.goodfiles== 'all': #Using all the files in the expected folder. if not, use the selected datasets
            self.datafilenames = np.sort(glob(self.homefolder + "/data/*-07-2019-stabilitystudies*.txt"))  # folder in which it is supposed to be stored the accelerometers data files
            self.sizeofgoodfiles = np.size(self.datafilenames)
        else:
            self.sizeofgoodfiles = np.size(self.goodfiles)
            self.day = [self.goodfiles[file][-6:-4] for file in range(self.sizeofgoodfiles)]
            self.month = [self.goodfiles[file][-8:-6] for file in range(self.sizeofgoodfiles)]
            self.year = [self.goodfiles[file][-12:-8] for file in range(self.sizeofgoodfiles)]
            self.datafilenames = np.sort([self.homefolder + '/data/' + self.day[file] + '-' + self.month[file] + '-' + self.year[file] + '-stabilitystudies-5files.txt' for file in range(self.sizeofgoodfiles)])  # DD-MM-YY

    def sample(self):
        for datafilenum in range(self.sizeofgoodfiles):
            justfilename = os.path.basename(self.datafilenames[datafilenum])
            print('Running OMA for file:',justfilename)
            OMAinscene = onefileanalysis.OMA(justfilename)
            OMAinscene.rawdataplot18()
            frequencyrange, yaxis, left, right = OMAinscene.FDD(desiredmaxfreq=10, numoflinestoplot=1, sensors=[9,10,11,15,16,17])
            peaks = OMAinscene.peakpicking()
            MACmatrix = OMAinscene.MACfunction()
            OMAinscene.MACplot(MACmatrix)
            newmodalfreq, goodindexinfullrange, newMACmatrix = OMAinscene.MACselection(MACmatrix)
            OMAinscene.MACplot(newMACmatrix)
            OMAinscene.enhancedfdd()
            OMAinscene.calibrate()
            OMAinscene.sensorshift()

class Track(object):

    #This class analyses the results of the OMA for a series of files, study trends, track modes and detect damage
    def __init__(self, **kwargs):
        self.MAClimit = 0.95
        self.goodfiles = kwargs.get('files', 'all')
        self.homefolder = os.path.normpath(os.getcwd()) #home directory of the system

        if type(self.goodfiles) == str and self.goodfiles== 'all': #Using all the files in the expected folder. if not, use the selected datasets
            self.EFDD_modalfreq_filenames = np.sort(glob(self.homefolder + "/output/onefileanalysis/analysisresults/*EFDD-modalfreq*.txt")) #folder in which it is supposed to be stored the accelerometers data files
            self.EFDD_modalshape_filenames = np.sort(glob(self.homefolder + "/output/onefileanalysis/analysisresults/*EFDD-modalshape*.txt")) #folder in which it is supposed to be stored the accelerometers data files
            self.shifts_filenames = np.sort(glob(self.homefolder + "/output/onefileanalysis/analysisresults/*Shift*.txt"))
        else:
            self.sizeofgoodfiles = np.size(self.goodfiles)
            self.day = [self.goodfiles[file][-6:-4] for file in range(self.sizeofgoodfiles)]
            self.month = [self.goodfiles[file][-8:-6] for file in range(self.sizeofgoodfiles)]
            self.year = [self.goodfiles[file][-12:-8] for file in range(self.sizeofgoodfiles)]
            self.datafilenames = np.sort([self.homefolder + "/data/" + self.day[file] + "-" + self.month[file] + "-" + self.year[file] + "-stabilitystudies-5files.txt" for file in range(self.sizeofgoodfiles)])  # DD-MM-YY
            self.EFDD_modalshape_filenames = np.sort(([self.homefolder + "/output/onefileanalysis/analysisresults/" + self.day[file] + "-" + self.month[file] + "-" + self.year[file] + "-stabilitystudies-5files-EFDD-modalshapes10.0Hzdec-.txt" for file in range(self.sizeofgoodfiles)]))
            self.EFDD_modalfreq_filenames = np.sort(([self.homefolder + "/output/onefileanalysis/analysisresults/" + self.day[file] + "-" + self.month[file] + "-" + self.year[file] + "-stabilitystudies-5files-EFDD-modalfreqs-dampratio10.0Hzdec-.txt" for file in range(self.sizeofgoodfiles)]))
            self.shifts_filenames = []
            for file in range(self.sizeofgoodfiles):
                filenamechannel = glob(self.homefolder + "/output/onefileanalysis/analysisresults/" + self.day[file] + "-" + self.month[file] + "-" + self.year[file] + '-stabilitystudies-5filesShift*.txt')
                self.shifts_filenames.append(np.sort(filenamechannel))

        self.numberofdatafiles = np.size(self.EFDD_modalfreq_filenames)
        self.numberofmodesdatafiles = np.size(self.EFDD_modalshape_filenames)
        self.date = [os.path.basename(self.EFDD_modalfreq_filenames[eachdate])[:10] for eachdate in range(self.numberofdatafiles)]

        #Seting directories
        self.resultsfolder = '/output/multiplefileanalysis/'
        try:
            os.makedirs(self.homefolder + self.resultsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def Trendanalysis(self):
        self.trackfreq = np.zeros(self.numberofdatafiles)
        self.trackdamping = np.zeros(self.numberofdatafiles)

        #Creating plots
        #Tracking modal freq
        figmodalfreq = plt.figure(figsize=(15, 8))
        xymodalfreq = figmodalfreq.add_subplot(111)
        #Tracking damping ratio
        figdamping = plt.figure(figsize=(15, 8))
        xydamping = figdamping.add_subplot(111)
        #Tracking mode shape
        figmodeshape = plt.figure(figsize=(15, 8))
        xymodeshape = figmodeshape.add_subplot(111)
        #Correlation between frequency and damping
        figfreqxdamping = plt.figure(figsize=(15, 8))
        freqxdamping = figfreqxdamping.add_subplot(111)


        self.EFDD_modalfreq, self.EFDD_modalshape = None, None
        self.numofmodesperfile = np.zeros(self.numberofdatafiles, dtype=int)
        self.numofchannelsindatafile = 6

        for filenum in range(self.numberofdatafiles):

            #Read and store information about modal frequencies and mode shapes
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

            freqxdamping.scatter(newfilefreq[0],100*newfilefreq[1])
            freqxdamping.set_xlabel('Frequency (Hz)', fontsize=15)
            freqxdamping.set_ylabel('Damping(%)', fontsize=15)
            freqxdamping.set_title('Frequency x Damping', fontsize=15)

            xymodalfreq.scatter(np.ones(self.numofmodesperfile[filenum])*filenum, newfilefreq[0], color='red')
            xymodalfreq.set_ylabel('Frequency(Hz)', fontsize=15)
            xymodalfreq.set_title('Modal Frequencies along the days', fontsize=15)

            xydamping.scatter(np.ones(self.numofmodesperfile[filenum])*filenum, 100*newfilefreq[1], color='red')
            xydamping.set_ylabel('Damping (%)', fontsize=15)
            xydamping.set_title('Damping along the days', fontsize=15)

            for freq in range(self.numofmodesperfile[filenum]):
                xymodeshape.scatter(np.ones(self.numofchannelsindatafile)*filenum, self.EFDD_modalshape[filenum][freq], marker='+', color='blue')
            xymodeshape.set_ylabel('Amplitude (a.u.)', fontsize=15)
            xymodeshape.set_xlabel('Channel', fontsize=15)
            xymodeshape.set_title('Mode shape along the days', fontsize=15)

        self.locs = np.arange(0,self.numberofdatafiles,1)
        xymodalfreq.set_xticks(ticks = self.locs)
        xydamping.set_xticks(ticks=self.locs)
        xymodalfreq.set_xticklabels(labels=self.date, rotation=90)
        xydamping.set_xticklabels(labels=self.date, rotation=90)

        # Calculate MAC value between modal frequencies for consecutive days
        self.MAC = np.zeros((self.numberofdatafiles - 1, np.amax(self.numofmodesperfile), np.amax(self.numofmodesperfile)))
        for filenum in range(self.numberofdatafiles - 1):
            for peak1 in range(self.numofmodesperfile[filenum]):
                for peak2 in range(self.numofmodesperfile[filenum+1]):
                    self.MAC[filenum, peak1, peak2] = np.square(np.dot(self.EFDD_modalshape[filenum][peak1], self.EFDD_modalshape[filenum + 1][peak2]))/(np.dot(self.EFDD_modalshape[filenum][peak1], self.EFDD_modalshape[filenum][peak1])*np.dot(self.EFDD_modalshape[filenum + 1][peak2],self.EFDD_modalshape[filenum + 1][peak2]))
                    #print(filenum,peak1,peak2,self.MAC[filenum, peak1, peak2])

        #Find which modal frequencies from subsequent days are correlated with one another
        self.correlatedfrequencies = np.array(np.where(self.MAC>self.MAClimit))


        #Count correlations per day
        self.numofcorrelations = np.size(self.correlatedfrequencies, axis=-1)
        self.correlperday = np.zeros(self.numberofdatafiles - 1, dtype=int)
        for filecounter in range(self.numberofdatafiles - 1):
            self.correlperday[filecounter] = np.size(np.where(self.correlatedfrequencies[0] == filecounter))

        #Defining the change factor (variable to monitor) - now in general form, later monitor each specific mode
        self.freq1, self.freq2 = [np.zeros(self.numofcorrelations, dtype=int) for i in range(2)]
        self.modeshapeshift = np.zeros((self.numofcorrelations,self.numofchannelsindatafile))
        self.freqshift, self.dampingshift = [np.zeros(self.numofcorrelations) for i in range(2)]
        self.changefactorfrequency, self.changefactordamping = [np.zeros((self.numberofdatafiles - 1)) for i in range(2)]

        for correlationum in range(self.numofcorrelations):
            self.whichfile = (self.correlatedfrequencies[0][correlationum]).astype(int)
            whichfileback = (self.correlatedfrequencies[0][correlationum - 1]).astype(int)
            #Frequencies of the first and the subsequent day in which there is a correlation
            self.freq1[correlationum] = (self.correlatedfrequencies[1][correlationum]).astype(int)
            self.freq2[correlationum] = (self.correlatedfrequencies[2][correlationum]).astype(int)

            #Shifts in the modal parameters for the subsequent correlated days (%)
            self.modeshapeshift[correlationum] = 100*(self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]] - self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]])/self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]]
            self.freqshift[correlationum] = 100*(self.EFDD_modalfreq[self.whichfile+1][0][self.freq2[correlationum]] - self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]])/self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]]
            self.dampingshift[correlationum] = 100*(self.EFDD_modalfreq[self.whichfile+1][1][self.freq2[correlationum]] - self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]])/self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]]

            #Variables for the plot
            linex = [self.whichfile,self.whichfile+1]
            lineyfreq = [self.EFDD_modalfreq[self.whichfile][0][self.freq1[correlationum]], self.EFDD_modalfreq[self.whichfile+1][0][self.freq2[correlationum]]]
            lineydamping = [100*self.EFDD_modalfreq[self.whichfile][1][self.freq1[correlationum]], 100*self.EFDD_modalfreq[self.whichfile+1][1][self.freq2[correlationum]]]
            linefreq = mlines.Line2D(linex, lineyfreq, c='black')
            linedamping = mlines.Line2D(linex, lineydamping, c='black')
            #XYlinefreqdamp = mlines.Line2D(lineyfreq, lineydamping, c='black')
            xymodalfreq.add_line(linefreq)
            xydamping.add_line(linedamping)
            for dimension in range(self.numofchannelsindatafile):
                lineymodeshapes = [self.EFDD_modalshape[self.whichfile][self.freq1[correlationum]][dimension], self.EFDD_modalshape[self.whichfile + 1][self.freq2[correlationum]][dimension]]
                linemodeshapes = mlines.Line2D(linex, lineymodeshapes, c='black')
                xymodeshape.add_line(linemodeshapes)
            #freqxdamping.add_line(XYlinefreqdamp)

        #Plot number of correlation per file: it may also indicate when there is a huge change and therefore no correlation (must have wind selection turned on)
        figcorrelation = plt.figure(figsize=(15, 8))
        xycorrelation = figcorrelation.add_subplot(111)
        xycorrelation.plot(self.locs[:-1],self.correlperday)
        xycorrelation.set_title('Number of correlation between consecutive days', fontsize=15)
        xycorrelation.set_xticks(ticks = self.locs[:-1])
        xycorrelation.set_xticklabels(labels=self.date, rotation=90)
        xycorrelation.set_ylabel('Number of correlations', fontsize=15)
        figcorrelation.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-numofcorrelations.png')
        plt.close()

        # Change factor variables
        for filenum in range(self.numberofdatafiles - 1):#errado essa soma
            firstindex = np.where(self.correlatedfrequencies[0]==filenum)[0][0]
            self.changefactorfrequency[filenum] = np.sum(self.freqshift[firstindex:firstindex+self.correlperday[filenum]])/self.correlperday[filenum]
            self.changefactordamping[filenum] = np.sum(self.dampingshift[firstindex:firstindex+self.correlperday[filenum]])/self.correlperday[filenum]

        #Monitoring relative changes in the frequency
        figchangefactorfrequency = plt.figure(figsize=(15, 8))
        xychangefactorfrequency = figchangefactorfrequency.add_subplot(111)
        #Monitor relative changes in the damping ratio
        figchangefactordamping = plt.figure(figsize=(15, 8))
        xychangefactordamping = figchangefactordamping.add_subplot(111)
        #Monitor relative changes in the mode shapes
        figchangemodeshape = plt.figure(figsize=(15, 8))
        xychangemodeshape = figchangemodeshape.add_subplot(111)

        xychangefactorfrequency.plot(self.locs[:-1],self.changefactorfrequency) #locs[:-1] = range(self.numofdatafiles - 1)
        xychangefactorfrequency.set_ylabel('Frequency deviation (%)', fontsize=15)
        xychangefactorfrequency.set_title('Frequency Shift for correlated modes', fontsize=15)
        xychangefactorfrequency.set_xticks(ticks = self.locs[:-1])
        xychangefactorfrequency.set_xticklabels(labels=self.date, rotation=90)

        xychangefactordamping.plot(range(self.numberofdatafiles - 1),self.changefactordamping)
        xychangefactordamping.set_ylabel('Damping deviation (%)', fontsize=15)
        #xychangefactordamping.set_xlabel('Correlation', fontsize=15)
        xychangefactordamping.set_title('Change factor in Damping for correlated modes', fontsize=15)
        xychangefactordamping.set_xticks(ticks = self.locs[:-1])
        xychangefactordamping.set_xticklabels(labels=self.date, rotation=90)

        if type(self.goodfiles) == str and self.goodfiles == 'all':
            figchangefactorfrequency.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-changefactorfreq.png')
            figchangefactordamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-changefactordamping.png')
            figmodalfreq.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-trackfreq.png')
            figdamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-trackdamping.png')
            figfreqxdamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-freqxdamping.png')
            figmodeshape.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-modeshapes.png')

        else:
            figchangefactorfrequency.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-changefactorfreq.png')
            figchangefactordamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-changefactordamping.png')
            figmodalfreq.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-trackfreq.png')
            figdamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-trackdamping.png')
            figfreqxdamping.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-freqxdamping.png')
            figmodeshape.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-modeshapes.png')
        plt.close()

        #Chi square indicator
        chisquarefreq, chisquaredamping, chisquaremodeshape = [np.zeros((self.numberofdatafiles-1)) for i in range(3)]
        for filenum in range(self.numberofdatafiles-1):
            squarefreq, squaredamp, squaremodeshape = [], [], []
            firstindex = np.where(self.correlatedfrequencies[0] == filenum)[0][0]
            for mode in range(self.correlperday[filenum]):
                squarefreq.append(np.square(self.freqshift[firstindex + mode]/100)*(self.EFDD_modalfreq[filenum][0][self.freq1[firstindex+mode]]))
                squaredamp.append(np.square(self.dampingshift[firstindex + mode]/100)*(self.EFDD_modalfreq[filenum][1][self.freq1[firstindex+mode]]))
                squaremodeshape.append(np.square(self.modeshapeshift[firstindex + mode] / 100) * (self.EFDD_modalshape[filenum][self.freq1[firstindex + mode]]))

            chisquarefreq[filenum] = np.sum(squarefreq)/self.correlperday[filenum]
            chisquaredamping[filenum] = np.sum(squaredamp)/self.correlperday[filenum]
            chisquaremodeshape[filenum] = np.sum(squaremodeshape)/(self.correlperday[filenum]*self.numofchannelsindatafile)

        figchifreq = plt.figure(figsize=(15, 8))
        xychifreq = figchifreq.add_subplot(111)
        xychifreq.plot(self.locs[:-1],chisquarefreq)
        xychifreq.set_ylabel(r'$\chi ^2$', fontsize=15)
        xychifreq.set_title(r'$\chi ^2$ for modal frequencies', fontsize=15)
        xychifreq.set_xticks(ticks = self.locs[:-1])
        xychifreq.set_xticklabels(labels=self.date, rotation=90)

        figchidamp = plt.figure(figsize=(15, 8))
        xychidamp = figchidamp.add_subplot(111)
        xychidamp.plot(self.locs[:-1],chisquaredamping)
        xychidamp.set_ylabel(r'$\chi ^2$', fontsize=15)
        xychidamp.set_title(r'$\chi ^2$ for damping ratio', fontsize=15)
        xychidamp.set_xticks(ticks = self.locs[:-1])
        xychidamp.set_xticklabels(labels=self.date, rotation=90)

        figchimodeshape = plt.figure(figsize=(15, 8))
        xychimodeshape = figchimodeshape.add_subplot(111)
        xychimodeshape.plot(self.locs[:-1],chisquaremodeshape)
        xychimodeshape.set_ylabel(r'$\chi ^2$', fontsize=15)
        xychimodeshape.set_title(r'$\chi ^2$ for mode shape', fontsize=15)
        xychimodeshape.set_xticks(ticks = self.locs[:-1])
        xychimodeshape.set_xticklabels(labels=self.date, rotation=90)

        if type(self.goodfiles) == str and self.goodfiles == 'all':
            figchifreq.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-chisquarefreq.png')
            figchidamp.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-chisquaredamp.png')
            figchimodeshape.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-allfiles-chisquaremodeshape.png')

        else:
            figchifreq.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-chisquarefreq.png')
            figchidamp.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-chisquaredamp.png')
            figchimodeshape.savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-chisquaremodeshape.png')
        plt.close()

        #Shifts between sensors (all values)
        figshift1,figshift2,figshift3 = [plt.figure(figsize=(15, 8)) for i in range(3)]
        xyshift1,xyshift2,xyshift3 = figshift1.add_subplot(111), figshift2.add_subplot(111), figshift3.add_subplot(111)
        plotdic = {0:xyshift1,1:xyshift2,2:xyshift3}
        figdict = {0:figshift1,1:figshift2,2:figshift3}
        titles = ['Sensor 1 - Sensor 2','Sensor 1 - Sensor 3', 'Sensor 2 - Sensor 3']
        labels = ['X axis','Y axis','Z axis']
        color= ['blue', 'red', 'green']
        nummaxofcorrel = 1 #change to 3 when third sensor works fine
        self.shifts_filenames = np.array(self.shifts_filenames)
        self.samplingratio = 100
        for filenum in range(self.numberofdatafiles):
            for correlation in range(nummaxofcorrel):
                newfileshift = np.loadtxt(self.shifts_filenames[filenum*nummaxofcorrel])
                sizeoffile = np.size(newfileshift, axis=1)
                time = np.linspace(filenum*sizeoffile/self.samplingratio, sizeoffile*(filenum+1)/self.samplingratio, sizeoffile)
                for dimension in range(3):
                    plotdic[correlation].plot(time[1:], np.power(10,6)*newfileshift[dimension,1:], color=color[dimension], label=labels[dimension])
                    plotdic[correlation].set_title('Shift between: ' + titles[correlation], fontsize=15)
                    plotdic[correlation].set_xticklabels(labels=self.date, rotation=90)
                    plotdic[correlation].set_xticks(ticks=self.locs * sizeoffile / self.samplingratio)
                    plotdic[correlation].set_ylabel(r'Shift ($\mu$m)', fontsize=15)
        for correlation in range(nummaxofcorrel):
            plotdic[correlation].grid()
            plotdic[correlation].legend(labels)
            figdict[correlation].savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-shift' + titles[correlation] +'.png')
        plt.close()

        # Shifts between sensors (day average)
        averagex, averagez, averagez = [],[],[]
        averagedict = {0:averagex, 1:averagez, 2:averagez}
        for filenum in range(self.numberofdatafiles):
            for correlation in range(nummaxofcorrel):
                newfileshift = np.loadtxt(self.shifts_filenames[filenum*nummaxofcorrel])
                for dimension in range(3):
                    averagedict[dimension] = np.append(averagedict[dimension],np.mean(newfileshift[dimension,:]))

        figshift1,figshift2,figshift3 = [plt.figure(figsize=(15, 8)) for i in range(3)]
        xyshift1,xyshift2,xyshift3 = figshift1.add_subplot(111), figshift2.add_subplot(111), figshift3.add_subplot(111)
        plotdic = {0:xyshift1,1:xyshift2,2:xyshift3}
        figdict = {0:figshift1,1:figshift2,2:figshift3}
        for dimension in range(3):
            plotdic[correlation].plot(self.locs, np.power(10,6)*averagedict[dimension], color=color[dimension], label=labels[dimension])
            plotdic[correlation].set_title('Shift between: ' + titles[correlation], fontsize=15)
            plotdic[correlation].set_xticklabels(labels=self.date, rotation=90)
            plotdic[correlation].set_xticks(ticks=self.locs)
            plotdic[correlation].set_ylabel(r'Shift ($\mu$m)', fontsize=15)
        for correlation in range(nummaxofcorrel):
            plotdic[correlation].grid()
            plotdic[correlation].legend(labels)
            figdict[correlation].savefig(self.homefolder + self.resultsfolder + self.date[0] + 'until' + self.date[-1] + '-selectedfiles-shift' + titles[correlation] +'-avg.png')
        plt.close()

class Weather(object):
#This class analyses the weather data to give a quality check for the accelerometers data sets

    def __init__(self):
        self.homefolder = os.path.normpath(os.getcwd()) #home directory of the system
        #self.homefolder = os.path.normpath(os.getcwd() + os.sep + os.pardir) #home directory of the system
        self.weatherdatafolder = self.homefolder + "/data/weather" #folder in which it is supposed to be stored the weather data files
        self.weatherdatafiles = np.sort(glob(self.weatherdatafolder + "/weather*.txt"))
        self.numberofdatafiles = np.size(self.weatherdatafiles)
        self.date = [os.path.basename(self.weatherdatafiles[eachdate])[11:-4] for eachdate in range(self.numberofdatafiles)]

        #Seting directories
        self.resultsfolder = '/output/multiplefileanalysis/weather/'
        try:
            os.makedirs(self.homefolder + self.resultsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def Analysis(self,**kwargs):
        self.timeofacquisition = kwargs.get('timeofacquisition', (0,24))
        self.windmeanthreshold = kwargs.get('windmeanthreshold', 1.1)
        self.winddirminvariance = kwargs.get('winddirminvariance', 180)
        self.numofvariables = 8
        self.Flags = np.zeros((self.numberofdatafiles,2))
        self.meanvalues, self.variance, self.minvalues, self.maxvalues = [np.zeros((self.numberofdatafiles,self.numofvariables)) for i in range(4)]

        for filenum in range(self.numberofdatafiles):
            print("Generating graphs for Weather:")
            print(self.weatherdatafiles[filenum])
            weatherclass = onefileanalysis.Weather(os.path.basename(self.weatherdatafiles[filenum]))
            weatherclass.Analysis(timeofacquisition=self.timeofacquisition)
            self.meanvalues[filenum], self.variance[filenum], self.minvalues[filenum], self.maxvalues[filenum] = weatherclass.Statistics()
            self.Flags[filenum] = weatherclass.Qualitycheck(windmeanthreshold=self.windmeanthreshold,winddirminvariance=self.winddirminvariance)
        return self.Flags

    def Selectionfiles(self):
        self.goodfileindexes = np.where((self.Flags[:,0]==1) & (self.Flags[:,1]==1))
        self.goodfileindexes=self.goodfileindexes[0]
        self.goodfiles = self.weatherdatafiles[self.goodfileindexes]
        return self.goodfiles

#class Reports(object):
'''This class sends reports and alerts based on the time series analysis'''