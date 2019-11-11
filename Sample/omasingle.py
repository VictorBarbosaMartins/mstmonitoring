import errno
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('agg')
import numpy as np
from scipy import signal
from sklearn import preprocessing
from scipy.optimize import curve_fit
import definitions as names


class OMA(object):
    '''This class analyses the datafile from the accelerometers, generates graphs and txt files with results

    #Suggested name for the datafiles (supposed to be stored at ./data/:
    #YYYY-MM-DD-MST-XX-STR.txt, where XX indicates the telescope

    Default Sampling ratio is 100 Hz;'''

    def __init__(self, filename):
        self.filename = filename
        # self.path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        self.path = os.environ["MST-STR-MON-DATA-CONVERTED"] + '/'

        self.samplingratio = 100  # Samp. ratio = 100 Hz. You can change it, but it should be a defined value forever
        # self.Victorinput = np.loadtxt(self.path + '/data/' + self.filename, delimiter=' ', dtype=float)
        self.Victorinput = np.loadtxt(self.path + self.filename, delimiter=' ', dtype=float)
        # Number of sample points
        self.N = np.size(self.Victorinput, axis=0)
        self.dt = 1 / self.samplingratio

        # Number of valid channels
        self.numofchannels = np.size(self.Victorinput, axis=1) - 6  # because now we have 6 inclinometers
        # remove it once we have no inclinometers anymore

        # Nyquist frequency
        self.nyquist = self.samplingratio / 2
        # Time during data acquisition
        self.time = np.linspace(0, self.N * self.dt, self.N)

        # Create directories to store results
        self.rawdatafolder = os.environ["MST-STR-MON-RAWDATAPLOTS"] + '/'
        self.resultsfolder = os.environ["MST-STR-MON-ONEFILEOMA"] + '/'
        createdirectories = [self.rawdatafolder, self.resultsfolder]
        self.filenamecomplete = self.path + filename

        for directory in createdirectories:
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def delete_results(self):
        self.OUT = self.rawdatafolder + self.filename[:-4] + names.RAW_ALL + '.png'
        self.OUTFDDFIG = self.resultsfolder + self.filename[:-4] + names.FDD + '.png'
        for string in [self.OUT, self.OUTFDDFIG]:
            try:
                #Delete results so the analysis can be ran again.
                os.remove(string)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
        return 0

    def rawdataplot18(self):
        '''Plot 18 channels in one window.
        to do: Remove/alter function once the number of available channels changes'''

        filename = self.filename
        self.OUT = self.rawdatafolder + self.filename[:-4] + names.RAW_ALL + '.png'

        if os.path.isfile(self.OUT):
            print("File " + self.OUT + " already exists.")
            self.DONTRUNFLAG = 1
        else:
            self.DONTRUNFLAG = 0
            Victorinput = self.Victorinput
            time = np.linspace(0, self.N * self.dt, self.N)
            f0 = plt.figure(figsize=(15, 8))
            print(self.numofchannels)
            self.correctedorder = np.array([15, 16, 17, 12, 13, 23, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6])
            xyi = np.zeros((self.numofchannels), dtype=object)
            counter = 0
            for sensor in self.correctedorder:
                xyi[counter] = f0.add_subplot(6, 3, counter + 1)
                xyi[counter].plot(time, Victorinput[:, sensor], linewidth=0.5)
                if sensor not in [16, 17]:
                    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='off', right='off',
                                    left='off', labelleft='off')
                else:
                    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='off', right='off',
                                    left='off', labelleft='off')
                if sensor in [0, 3, 6, 9, 12]:
                    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='off', right='off',
                                    left='on', labelleft='on')
                if sensor == 15:
                    plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off',
                                    left='on', labelleft='on')
                counter = counter + 1
            xyi[1].set_title('Raw data', fontsize=15)
            xyi[6].set_ylabel('Acceleration (mA)', fontsize=15)
            xyi[16].set_xlabel('Time(s)', fontsize=15)
            f0.savefig(self.OUT)
            plt.close(f0)
        return self.DONTRUNFLAG

    def rawdataplot(self, channel):
        '''Plot the raw data for one channel from 1 to number max. of channels'''
        self.OUT = self.rawdatafolder + self.filename[:-4] + names.RAW_CH + str(self.channel) + '.png'
        if os.path.isfile(self.OUT):
            print("File " + self.OUT + " already exists.")
            DONTRUNFLAG = 1
        else:
            DONTRUNFLAG = 0
            f0 = plt.figure(figsize=(15, 8))
            xyi = f0.add_subplot(111)
            xyi.plot(self.time, self.Victorinput[:, self.channel - 1], linewidth=0.5)
            xyi.set_title('Raw data', fontsize=15)
            xyi.set_ylabel('Acceleration (mA)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid()
            xyi.set_xlabel('Time(s)', fontsize=15)
            f0.savefig(self.OUT)
            plt.close(f0)
        return DONTRUNFLAG

    def calibrate(self, **kwargs):
        self.voltagerange = kwargs.get('voltagerange', np.array([4, 20]))  # in mA (after V-mA convertion)
        self.gravityrange = kwargs.get('gravityrange', np.array([-2, 2]))  # in g units
        self.correctedordergoodsensors = [12, 13, 23, 10, 11, 0, 4, 5, 6]
        self.sensors = kwargs.get('sensors', np.array(self.correctedordergoodsensors))
        self.numofchannelsnew = np.size(self.sensors)
        self.Calibrateddata = np.zeros((np.size(self.Victorinput, axis=0), self.numofchannelsnew))

        voltagerange = kwargs.get('voltagerange', np.array([4, 20]))  # in mA (after convertion)
        gravityrange = kwargs.get('gravityrange', np.array([-2, 2]))  # in g units

        # Tilts in the flanges (resulted from the elevation scan)
        turnaroundxaxis = np.pi / 180 * np.array([1.15, -0.24, 4.19])
        turnaroundyaxis = np.pi / 180 * np.array([-0.24, 0.46, 0.58])
        turnaroundzaxis = np.pi / 180 * np.array([4.19, -8.72, 0])

        # For z channels add 1 g

        for xaxis in [0, 3, 6]:
            self.Calibrateddata[:, xaxis] = (
                    np.diff(gravityrange) * (self.Victorinput[:, self.correctedordergoodsensors[xaxis]] - 12) / np.diff(
                voltagerange))
        for yaxis in [1, 4, 7]:
            self.Calibrateddata[:, yaxis] = (
                    np.diff(gravityrange) * (self.Victorinput[:, self.correctedordergoodsensors[yaxis]] - 12) / np.diff(
                voltagerange))
        for zaxis in [2, 5, 8]:
            self.Calibrateddata[:, zaxis] = (
                    np.diff(gravityrange) * (self.Victorinput[:, self.correctedordergoodsensors[zaxis]] - 12) / np.diff(
                voltagerange) + 1)

        for sensor in range(self.numofchannelsnew // 3):
            xaxis = self.Calibrateddata[:, sensor * 3]
            yaxis = self.Calibrateddata[:, sensor * 3 + 1]
            zaxis = self.Calibrateddata[:, sensor * 3 + 2]
            costheta = np.cos(turnaroundxaxis[sensor])
            sintheta = np.sin(turnaroundxaxis[sensor])
            cosphi = np.cos(turnaroundyaxis[sensor])
            sinphi = np.sin(turnaroundyaxis[sensor])
            cosbeta = np.cos(turnaroundzaxis[sensor])
            sinbeta = np.sin(turnaroundzaxis[sensor])
            # One rotation in Y, one rotation in X, one rotation in Z
            self.Calibrateddata[:, sensor * 3:(sensor + 1) * 3] = np.array([
                cosbeta * xaxis * cosphi + cosbeta * sinphi * zaxis + sinbeta * costheta * yaxis + sinbeta * sintheta * (
                        -sinphi * xaxis + cosphi * zaxis),
                -xaxis * cosphi * sinbeta - sinbeta * sinphi * zaxis + cosbeta * costheta * yaxis + cosbeta * sintheta * (
                        -sinphi * xaxis + cosphi * zaxis),
                (-sintheta) * yaxis + costheta * (
                        -sinphi * xaxis + cosphi * zaxis)]).transpose()

        # For sensors on CSS and dish invert Y and Z
        for channels in [1, 2, 7, 8]:  # Y and Z for each sensor
            self.Calibrateddata[:, channels] = - self.Calibrateddata[:, channels]
        self.Victorinput[:, self.correctedordergoodsensors] = self.Calibrateddata
        return self.Calibrateddata

    def FDD(self, **kwargs):

        '''
        Applies the FDD method to the accelerometers data and returns the modal frequencies and modal shapes.
        :param kwargs:
        desiredmaxfreq: Maximum frequency to monitor in Hz (10);
        resolution: Number of bins in Frequency range (2048);
        datapoints: Quantity of data to be considered (all datapoints - int);
        numoflinestoplot: number of singular value lines (SVD) to plot (1);
        self.sensors: list of channels to be taken into account (from channel 0 to max. number of channel - 1)
        :return: frequency range, Singular value spectral power in dB, Left side modal shape matrix,
        Right side modal shape matrix
        '''
        # self.sensors = kwargs.get('selectchannels', self.sensors)  # in g units
        self.desiredmaxfreq = kwargs.get('desiredmaxfreq', 10)
        self.resolution = kwargs.get('resolution', 2048 * 2)
        self.datapoints = kwargs.get('datapoints', self.N)
        self.numoflinestoplot = kwargs.get('numoflinestoplot', 3)
        # self.sensors = kwargs.get('sensors', np.arange(0, self.numofchannels))
        numberoffreqlines = np.round(self.resolution / 2).astype(int) + 1
        percentagetooverlap = 0.66
        # factorofdecimation = int(self.nyquist / self.desiredmaxfreq - self.nyquist % self.desiredmaxfreq)
        factorofdecimation = int(np.around(self.nyquist / self.desiredmaxfreq))
        outputresolution = self.N / factorofdecimation
        self.freal = np.around(self.nyquist / factorofdecimation,
                               1)  # shifted from desiredmaxfreq such as the factor of decimation become an integer
        sampleratingdecimated = 2 * self.freal
        timedecimated = np.linspace(0, self.N * self.dt, outputresolution)
        xyi = np.zeros((self.numofchannelsnew), dtype=object)
        self.Victordecimated = []
        self.Victorspectraldensity = np.zeros((numberoffreqlines, self.numofchannelsnew, self.numofchannelsnew))
        self.Victorsingularvalues = np.zeros((numberoffreqlines, self.numofchannelsnew))
        self.left, self.right = [
            np.zeros((numberoffreqlines, self.numofchannelsnew, self.numofchannelsnew)) for i in range(2)]

        self.OUTFDDFIG = self.resultsfolder + self.filename[:-4] + names.FDD + '.png'

        self.DONTRUNFLAG = 0

        if os.path.isfile(self.OUTFDDFIG):
            self.DONTRUNFLAG = 1
            print("File " + self.OUTFDDFIG + " already exists.")
            print("If you wish to run analysis, please delete results")
            return 0, 0, 0, 0, self.DONTRUNFLAG
        else:

            for sensor in self.sensors:
                # SCALING/DETRENDING
                preprocessing.scale(self.Victorinput[:, sensor], with_mean=True, with_std=False, copy=False)
                # std must be false! Each sensor has its own sensitivity, they are not equal

                # DECIMATING
                self.Victordecimated.append(signal.decimate(self.Victorinput[:self.N, sensor], factorofdecimation))
            self.Victordecimated = np.array(self.Victordecimated).transpose()

            # CROSS SPECTRAL DENSITY (CSD)
            for sensor1 in range(self.numofchannelsnew):
                for sensor2 in range(self.numofchannelsnew):
                    self.frequencyrange, self.Victorspectraldensity[:, sensor1, sensor2] = signal.csd(
                        self.Victordecimated[:, sensor1],
                        self.Victordecimated[:, sensor2],
                        fs=sampleratingdecimated,
                        nperseg=self.resolution,
                        noverlap=int(
                            percentagetooverlap * self.resolution),
                        detrend=False)
            # SVD
            self.reference = np.power(10, 1 / 10)  # Refers to 1 dB
            for frequencyline in range(numberoffreqlines):
                self.left[frequencyline], self.Victorsingularvalues[frequencyline], self.right[
                    frequencyline] = np.linalg.svd(
                    self.Victorspectraldensity[frequencyline].reshape(self.numofchannelsnew, self.numofchannelsnew),
                    full_matrices=True,
                    compute_uv=True)
                # Proof that A = u.s.u
                # np.allclose(self.Victorspectraldensity[frequencyline].reshape(self.numofchannelsnew, self.numofchannelsnew), np.dot(self.left[frequencyline]*self.Victorsingularvalues[frequencyline],self.right[frequencyline]))

            # Plot
            fig = plt.figure(figsize=(12, 8))
            xyi = fig.add_subplot(111)
            self.singvaluesindecb = np.zeros((numberoffreqlines, self.numoflinestoplot))
            for sensor in range(self.numoflinestoplot):
                self.singvaluesindecb[:, sensor] = 10 * np.log10(self.Victorsingularvalues[:, sensor] / self.reference)
                xyi.plot(self.frequencyrange, self.singvaluesindecb[:, sensor], linewidth=1)
                # xyi.plot(frequencyrange,Victorsingularvalues[:,sensor],linewidth=1)
                # xyi.set_yscale('log')
            plt.xlabel('Frequency (Hz)', fontsize=15)
            # plt.ylabel('Intensity('+r'$(m/s^2)²/Hz)$'+')',fontsize=15)
            plt.ylabel('Intensity(dB)', fontsize=20)
            plt.xticks(np.arange(min(self.frequencyrange), max(self.frequencyrange) + 1, 1), fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid()
            plt.xlim(0, self.desiredmaxfreq)
            plt.ylim(-120, -30)
            plt.title('OMA spectrum', fontsize=15)
            fig.savefig(self.OUTFDDFIG)
            np.savetxt(self.resultsfolder + self.filename[:-4] + names.FDD + str(self.sensors) + '-' + str(
                np.around(self.freal, 1)) + 'Hzdec-singvalues.txt',
                       self.Victorsingularvalues)
            np.savetxt(self.resultsfolder + self.filename[:-4] + names.FDD + str(self.sensors) + '-' + str(
                np.around(self.freal, 1)) + 'Hzdec-frequencies.txt',
                       self.frequencyrange)
            # plt.clf()
            # plt.close(fig)

            # Estimating noise level
            self.noiselevelindb = np.mean(
                self.singvaluesindecb[int(0.1 * self.resolution // 2):int(0.8 * self.resolution // 2),
                self.numoflinestoplot - 1])
            self.noiselevel = np.mean(
                self.Victorsingularvalues[int(0.1 * self.resolution // 2):int(0.8 * self.resolution // 2),
                self.numoflinestoplot - 1])

            return self.frequencyrange, self.singvaluesindecb, self.left, self.right, self.DONTRUNFLAG

    def peakpicking(self, **kwargs):

        '''
        Search, find and return peaks in the FDD spectrum
        :param frequencyrange: Range of frequency (1-D numpy array)
        :param Victorsingularvalues: Singular values from the FDD method (1-D or higher order np array)
        :param kwargs:
        roi ((0.5,7.5)): region of interest (roi) in Hz; obs.: used to define minimum height of peak;
        numoflinestoplot (1): how many singular value 1D matrixes should the function search for peaks;
        inputheight (10): Minimum height of peak in dB in relation to ground;
        width (3): Minimum width for the peak;
        :return:peaksstorage: position of the peaks in the frequencyrange matrix.'''

        self.rangeofinterest = kwargs.get('roi', (0.5,
                                                  7.5))  # Freq. range to monitor the peaks, defined as the region where the modes tend to be easily excited (appear in every data taking)
        self.numoflinestoplot = kwargs.get('graphlines', self.numoflinestoplot)
        self.inputheight = kwargs.get('inputheight', 15)  # Minimum height of peak in dB in relation to ground
        self.distance = kwargs.get('distance', 10)  # Minimum distance between peaks
        self.width = kwargs.get('width', 3)  # Minimum width for the peak

        #size = np.size((self.singvaluesindecb), axis=0)
        numofchannelsforpeaks = np.size((self.singvaluesindecb), axis=1)
        self.peaksstorage = []
        numberoffreqlines = np.size(self.Victorsingularvalues, axis=0)

        for sensor in range(numofchannelsforpeaks):
            minindex = np.where(self.frequencyrange > self.rangeofinterest[0])[0][0]
            maxindex = np.where(self.frequencyrange < self.rangeofinterest[1])[0][-1]
            mininregionofinterest = np.amin(self.singvaluesindecb[minindex:maxindex, sensor])
            maxinregionofinterest = np.amax(self.singvaluesindecb[minindex:maxindex, sensor])
            height = mininregionofinterest + self.inputheight
            peaks, _ = signal.find_peaks(self.singvaluesindecb[:, sensor], height=height, width=self.width,
                                         distance=self.distance)
            self.peaksstorage.append(peaks)
        self.peaksstorage = np.array(self.peaksstorage)

        # PLOT PEAKS
        f9 = plt.figure(figsize=(12, 8))
        xyi = f9.add_subplot(111)
        # numberofpeaks = np.zeros((numofchannelsforpeaks), dtype=int)

        # for singvalueline in range(self.numoflinestoplot):
        # numberofpeaks = np.size(self.peaksstorage[0])  # number of peaks in each singular value line
        xyi.plot(self.frequencyrange, self.singvaluesindecb[:, 0], linewidth=1)
        # xyi.plot(frequencyrange,np.ones(size)*averageofinterest,linewidth=1, c='black', label='Mean {0} - {1} Hz'.format(self.rangeofinterest[0],self.rangeofinterest[1]))
        # xyi.set_yscale('log')
        xyi.set_xlim(0, 10)
        # xyi.fill_betweenx(Victorsingularvalues[:,sensor], frequencyrange[minindex], frequencyrange[maxindex],
        # facecolor='green', alpha=0.3)
        # indexfreqofinterest = np.linspace(minindex,maxindex, dtype=int)
        xyi.scatter(self.frequencyrange[self.peaksstorage[0]],
                    self.singvaluesindecb[self.peaksstorage[0], 0], marker='+', color='red',
                    label='Potential modal frequencies')
        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.ylim(-120, -30)
        # plt.ylabel('Intensity('+r'$(m^2/s)/Hz)$'+')', fontsize=15)
        plt.ylabel('Intensity(dB)', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(np.arange(min(self.frequencyrange), max(self.frequencyrange) + 1, 1), fontsize=15)
        plt.grid()
        plt.title('OMA with frequency peaks', fontsize=15)
        plt.legend(fontsize=15, loc=3)
        f9.savefig(self.resultsfolder + self.filename[:-4] + names.FDD + str(self.sensors) + names.PEAKS + '-' + str(
            np.around(self.freal, 1)) + names.DECIMATION + '.png')
        # plt.close(f9)
        # del f9
        return self.peaksstorage

    # MODAL ASSURANCE CRITERION (MAC) - FIRST CALCULATION
    def MACfunction(self):
        '''
        Calculate the MAC value for each possible pair of peak encountered in the peakpicking function.
        :param frequencyrange: Range of frequency (1-D numpy array);
        :param peaksstorage: Matrix which contains the position of the peaks in the frequencyrange;
        :param left: left singular vector resulted from the FDD method;
        :param right:  right singular vector resulted from the FDD method;
        :return: MACmatrix, matrix n x n with the MAC value for each pair of n encountered peaks.
        '''
        self.macselectionflag = 0  # indicate if a selection was already done for that estimated MAC values
        numberofpeaks = np.size(self.peaksstorage[0])  # number of peaks in each singular value line
        MACmatrix = np.zeros((numberofpeaks, numberofpeaks))  # MAC matrix for each singular value line
        peaknumber1 = 0
        for peak1 in self.peaksstorage[0]:
            peaknumber2 = 0
            for peak2 in self.peaksstorage[0]:
                MACmatrix[peaknumber1, peaknumber2] = np.square(
                    np.dot(self.left[peak1, :, 0], self.left[peak2, :, 0])) / (
                                                              np.dot(self.left[peak1, :, 0],
                                                                     self.left[peak1, :, 0]) * np.dot(
                                                          self.left[peak2, :, 0], self.left[peak2, :, 0]))
                peaknumber2 += 1
            peaknumber1 += 1
        return MACmatrix

    def MACplot(self, MAC):

        '''
        Plot the MAC value for the inputed peaks
        :param frequencyrange: Range of frequency (1-D numpy array)
        :param peakindexes: n-D numpy array which contains the position of the peaks in the frequencyrange for the singular value lines
        :param MAC: n-D numpy array which contains the MAC-values for the n x n combination of peaks
        :return: plot
        '''
        peakindexes = self.peaksstorage[0].astype(int)
        numberofpeaks = np.size(MAC, axis=0)
        f10 = plt.figure(figsize=(10, 8))
        xyi = f10.add_subplot(111)
        imagem = xyi.imshow(np.flip(MAC, axis=0), shape=(numberofpeaks, numberofpeaks), vmin=0, vmax=1)
        plt.locator_params(axis='both', nbins=numberofpeaks)
        if self.macselectionflag == 0:
            labelinfreq = np.round(self.frequencyrange[peakindexes], 2).flatten()
        elif self.macselectionflag == 1:
            labelinfreq = np.round(self.frequencyrange[self.goodindexinfullrange], 2).flatten()
        labelsx = [item.get_text() for item in xyi.get_xticklabels()]
        labelsy = [item.get_text() for item in xyi.get_yticklabels()]
        labelsx[1:] = labelinfreq
        labelsy[1:] = labelinfreq[::-1]
        xyi.set_xticklabels(labelsx, fontsize=15)
        xyi.set_yticklabels(labelsy, fontsize=15)
        plt.xticks(rotation='vertical')
        cbar = f10.colorbar(imagem)
        minvalue = np.amin(self.frequencyrange[peakindexes])
        maxvalue = np.amax(self.frequencyrange[peakindexes])
        xyi.set_xlabel('Freq.(Hz)', fontsize=15)
        xyi.set_ylabel('Freq.(Hz)', fontsize=15)
        xyi.set_title('MAC - Peaks', fontsize=15)
        if self.macselectionflag == 0:
            f10.savefig(self.resultsfolder + self.filename[:-4] + names.MACVALUE_ALL + '.png')
        elif self.macselectionflag == 1:
            f10.savefig(self.resultsfolder + self.filename[:-4] + names.MACVALUE_SELECTED + '.png')

        # plt.close(f10)
        # del f10
        return 0

    def MACselection(self, MAC, **kwargs):

        '''
        Select based on the MAC value the peaks which are most probably modal frequencies of the structure.
        :param frequencyrange: Range of frequency (1-D numpy array);
        :param peaksstorage: Matrix which contains the position of the peaks in the frequencyrange;
        :param MAC: n-D numpy array which contains the MAC-values for the n x n combination of peaks;
        :param kwargs(default):
        maclimit (0.15): limit MAC value to accept two modal shapes as linearly independent (LI);
        :return:
        newmodalfreq: frequencies of the peaks which passed the LI test (modal frequencies);
        goodindexinfullrange: indexes of the modal frequencies in the full range of available frequencies;
        newMACmatrix: new MAC matrix with calculated for the modal frequencies.
        '''

        # Delete linearly dependent potential modal frequencies
        # Preserving the pot. modal freq. with smaller Sum of MAC
        # If the Sum of MACs are equal, preserve the lower freq
        self.macselectionflag = 1  # turns the flag to one to show that a selection was already done for those MAC values
        self.maclimit = kwargs.get('maclimit', 0.7)
        numofpeaks = np.size(self.peaksstorage[0])
        includedindexestodelete = np.where((MAC > self.maclimit) & (MAC != 1))  # and !=1
        modalfreq = np.array([self.frequencyrange[self.peaksstorage[0]]])[0, :]
        storeindextodelete = []
        badcorrelation1 = includedindexestodelete[0]
        badcorrelation2 = includedindexestodelete[1]

        for element in range(np.size(badcorrelation1)):
            sumofmacs = np.array(
                [np.sum(MAC[badcorrelation1[element], :]), np.sum(MAC[badcorrelation2[element], :])])

            indexofmaxMAC = np.argmax(sumofmacs)
            # If the sum of MACs are the same, the largest frequency is discarted
            if sumofmacs[0] == sumofmacs[1]:
                deleteindex = np.amax([badcorrelation1[element], badcorrelation2[element]])
            # otherwise the frequency with the largest MAC value is discarted
            else:
                if indexofmaxMAC == 0:
                    deleteindex = badcorrelation1[element]
                if indexofmaxMAC == 1:
                    deleteindex = badcorrelation2[element]
            if np.size(storeindextodelete) == 0:
                storeindextodelete = np.array([deleteindex])
            else:
                storeindextodelete = np.append(storeindextodelete, deleteindex)

        newmodalfreq = np.delete(modalfreq, np.unique(storeindextodelete))
        newnumberofpeaks = np.size(newmodalfreq)
        goodindexinfullrange, goodindexinmodalfreq = [np.zeros((newnumberofpeaks), dtype=int) for i in range(2)]
        newMACmatrix = np.zeros((newnumberofpeaks, newnumberofpeaks))
        for newpeak1 in range(newnumberofpeaks):
            goodindexinfullrange[newpeak1] = np.array(np.where(self.frequencyrange == newmodalfreq[newpeak1])).flatten()
            goodindexinmodalfreq[newpeak1] = np.array(np.where(modalfreq == newmodalfreq[newpeak1])).flatten()
        for newpeak1 in range(newnumberofpeaks):
            for newpeak2 in range(newnumberofpeaks):
                newMACmatrix[newpeak1, newpeak2] = MAC[goodindexinmodalfreq[newpeak1], goodindexinmodalfreq[newpeak2]]
        self.goodindexinfullrange = np.reshape(goodindexinfullrange, (
            1, np.size(goodindexinfullrange)))  # to fit the same shape of self.peakstorage
        # print(newmodalfreq)
        return newmodalfreq, self.goodindexinfullrange, newMACmatrix

    def enhancedfdd(self, **kwargs):
        '''
        Estimate the modal frequencies and damping ratio based on the autocorrelation function (IDFT of the FDD singular values)
        frequencyrange:param: Range of frequency (1-D numpy array);
        goodindexinfullrange:param: indexes of the modal frequencies in the full range of available frequencies;
        left:param: Left side modal shape matrix
        maclimit:param: Minimum MAC value such that the shape relates to the same modal frequency
        :return:
        '''

        def gaussianfunction(x, a, mu, sigma):
            return a * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))

        self.enhancedmaclimit = kwargs.get('maclimit', 0.85)
        self.correlationroi = kwargs.get('correlationroi', (0.30, 0.95))

        self.efddmodalfreqs, self.efdddampingratio, self.enhancedmodalshape = [], [], []
        maxnumberofpeaksinfdd = np.size(self.goodindexinfullrange[0])

        for peak in self.goodindexinfullrange[0]:
            # storeindexes = [peak]
            # Selecting frequencies around the peak, which the the MAC in relation to the peak is above the MAC limit
            # Compare each DOF of the frequency lines with the mode shape of the peak
            freqforward = peak + 1
            freqbackward = peak - 1
            storeindexes = [[] for i in range(self.numoflinestoplot)]
            for line in range(0, self.numoflinestoplot):
                MACforward, MACbackward = 1., 1.
                while MACforward >= self.enhancedmaclimit or MACbackward >= self.enhancedmaclimit:
                    MACforward = np.square(np.dot(self.left[peak, :, 0], self.left[freqforward, :, line])) / (
                            np.dot(self.left[peak, :, 0], self.left[peak, :, 0]) * np.dot(
                        self.left[freqforward, :, line], self.left[freqforward, :, line]))
                    MACbackward = np.square(np.dot(self.left[peak, :, 0], self.left[freqbackward, :, line])) / (
                            np.dot(self.left[peak, :, 0], self.left[peak, :, 0]) * np.dot(
                        self.left[freqbackward, :, line], self.left[freqbackward, :, line]))

                    if freqforward < np.size(self.frequencyrange) and MACforward >= self.enhancedmaclimit:
                        storeindexes[line].append(freqforward)
                        freqforward = freqforward + 1

                    if freqbackward > 0 and MACbackward >= self.enhancedmaclimit:
                        storeindexes[line].append(freqbackward)
                        freqbackward = freqbackward - 1
                    # print(MACforward,MACbackward)
                    # print(freqforward,freqbackward)
                # storeindexes[line] = [line,np.unique(storeindexes)] #Storing the goodindexes with the respective svd line
            bellfunction = np.zeros(np.size(self.frequencyrange))
            # storeindexes = np.array(storeindexes)

            bellfunction[peak] = self.Victorsingularvalues[peak, 0]
            for svdline in range(self.numoflinestoplot):
                bellfunction[storeindexes[svdline]] = self.Victorsingularvalues[storeindexes[svdline], svdline]

            """for svdline in range(self.numoflinestoplot):
                try: # probably there is just one line of the svd which is used in the bell form and not all the lines we are working with
                    bellfunction[storeindexes[svdline*2+1]] = self.Victorsingularvalues[storeindexes[svdline*2+1], svdline]
                except:
                    continue"""

            # Plot the bell shape for the peak
            fig = plt.figure(figsize=(10, 8))
            xyi = fig.add_subplot(111)
            xyi.plot(self.frequencyrange, bellfunction, c='red', linewidth=3.5)
            xyi.set_xlabel('Freq.(Hz)', fontsize=15)
            # xyi.set_ylabel('Intensity (dB)', fontsize=15)
            xyi.set_title('Bell shape - peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))),
                          fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            xyi.set_xlim(self.frequencyrange[peak] - .25, self.frequencyrange[peak] + .25)
            fig.savefig(self.resultsfolder + self.filename[:-4] + names.BELLSHAPE + '-' + str(
                np.around(self.frequencyrange[peak], decimals=2)) + 'Hz.png')
            plt.close()
            # if self.frequencyrange[peak] > 1:

            '''#Verify if the Bell curve is a gaussian distribution
            def Gauss(x, a, x0, sigma):
                return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
            storeindexes[0] =  np.unique(storeindexes[0])
            print(storeindexes[0])
            print(bellfunction[storeindexes[0]])
            popt,pcov = curve_fit(Gauss, self.frequencyrange[storeindexes[0]], bellfunction[storeindexes[0]], p0=[max(bellfunction[storeindexes[0]]), np.mean(bellfunction[storeindexes[0]]), np.std(bellfunction[storeindexes[0]])])
            print("The covariance of the Bell function fit is:",pcov)
            #plt.plot(self.frequencyrange[storeindexes[0]],bellfunction[storeindexes[0]])
            #plt.plot(self.frequencyrange[storeindexes[0]],Gauss(self.frequencyrange[storeindexes[0]],popt[0],popt[1],popt[2]))
            #plt.show()'''

            # Derive the Correlation function from the bell curve
            correlationfunc = np.fft.ifft(np.append(bellfunction, bellfunction[
                                                                  -2::-1]).flatten())  # pluging together 0 , positive, negative terms, excluding the zero the second time
            correlationfunc = correlationfunc[:np.size(
                correlationfunc) // 2]  # it is an odd number, so the zero frequency value will be gone
            correlationfunc = correlationfunc.real - np.mean(correlationfunc.real)
            normcorrelationfunc = correlationfunc / np.amax(correlationfunc)
            time = np.linspace(0, np.size(self.frequencyrange) / (2 * self.desiredmaxfreq),
                               np.size(self.frequencyrange), endpoint=False)  # time = resolution/(2*Ny)

            # Selecting region of interest (the k's peaks within the ROI)
            correlationpeakspositive, _ = signal.find_peaks(normcorrelationfunc, distance=1)  # Finding maximum
            correlationpeaksnegative, _ = signal.find_peaks(-normcorrelationfunc, distance=1)  # Finding minimum
            correlationpeaks = np.append(correlationpeakspositive, correlationpeaksnegative)

            # correlationpeaks = correlationpeakspositive
            correlationpeaks = np.sort(np.array(correlationpeaks).flatten())

            # Start counting the good indexes whenever we have at least 3 consecutive peaks inside the ROI
            consecutivepeaks = 3
            goodindexes = []

            for indexinpeaks in range(np.size(correlationpeaks) - consecutivepeaks):
                atleast3peaksinroi = np.abs(
                    normcorrelationfunc[correlationpeaks[indexinpeaks:indexinpeaks + consecutivepeaks]])
                condition1 = np.all(atleast3peaksinroi >= self.correlationroi[0])
                condition2 = np.all(atleast3peaksinroi <= self.correlationroi[1])
                if (condition1 == True and condition2 == True):
                    goodindexes.append(indexinpeaks)
            goodindexes = np.unique(goodindexes).flatten()

            totalnumofpeaks = np.size(correlationpeaks)
            #totalnumofpeaksintheroi = np.size(correlationpeaks[goodindexes[0]:goodindexes[-1] + 1])

            if np.size(goodindexes) >= 3: #at least three data points in the ROI
            #for 30.09.2019 the first mode rapidly decay, and there is no point in this ROI
                numofthepeaks = np.linspace(1, totalnumofpeaks, totalnumofpeaks, endpoint=True, dtype=int)

                logdecfsitparam = np.polyfit(numofthepeaks[goodindexes],
                                             2 * np.log(np.abs(normcorrelationfunc[correlationpeaks[goodindexes]])), deg=1)

                fittedlogdec = np.polyval(logdecfsitparam, numofthepeaks[goodindexes])
                logdecfactor = logdecfsitparam[0]

                # Plot the LogDec
                fig3 = plt.figure(figsize=(10, 8))
                xyi3 = fig3.add_subplot(111)
                xyi3.scatter(time[correlationpeaks], 2 * np.log(np.abs(normcorrelationfunc[correlationpeaks])), marker='+',
                             label='data')
                xyi3.plot(time[correlationpeaks[goodindexes]], fittedlogdec, c='red', label='linear fitting')
                xyi3.set_xlabel('Time lag (s)', fontsize=15)
                xyi3.set_ylabel(r'$ ln|r_{k_{0}}/r_{k}|$', fontsize=15)
                xyi3.set_title('LogDec estimation - peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))),
                               fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.legend()
                fig3.savefig(self.resultsfolder + self.filename[:-4] + names.LOGDEC + '-' + str(
                    np.around(self.frequencyrange[peak], 2)) + 'Hz.png')
                plt.close()

                # Estimate damping factor from the LogDec factor
                # dampingfactor = logdecfactor/np.sqrt(np.square(logdecfactor) + 4*np.square(np.pi))
                dampingfactor = 1 / (np.sqrt(1 + np.square(2 * np.pi / logdecfactor)))
                print('dampingfactor', dampingfactor)

                '''def expdecay(x,a,b):
                    return a*np.exp(b*x)
                popt, pcov = curve_fit(expdecay,time[1+roipeaks[1:]],np.abs(normcorrelationfunc[roipeaks[1:]]), p0=[1,-0.01])
                print(popt)'''

                # Plot the Correlation function (damping curve)
                fig2 = plt.figure(figsize=(10, 8))
                xyi2 = fig2.add_subplot(111)
                xyi2.plot(time[1:], normcorrelationfunc, linewidth=2.5, label='Autocorr. function')
                xyi2.scatter(time[correlationpeaks[goodindexes]], normcorrelationfunc[correlationpeaks[goodindexes]],
                             marker='+', color='black')
                xyi2.plot(time[correlationpeaks[goodindexes[0]]:correlationpeaks[goodindexes[-1]]],
                          normcorrelationfunc[correlationpeaks[goodindexes[0]]:correlationpeaks[goodindexes[-1]]],
                          linewidth=3, c='red', label='ROI')
                # xyi2.plot(time[1+roi], 0.4+np.exp(-decayfactor*roi), linewidth=3, c='green', label='decay envelope')

                # xyi2.plot(time[1+roipeaks[1:]],expdecay(roipeaks[1:],popt[0],popt[1]))

                xyi2.set_xlabel('Time (s)', fontsize=15)
                xyi2.set_ylabel('Normalized intensity (a.u.)', fontsize=15)
                xyi2.set_title(
                    'Autocorrelation function for peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))),
                    fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                fig2.savefig(self.resultsfolder + self.filename[:-4] + names.AUTOCORREL + '-' + str(
                    np.around(self.frequencyrange[peak], 2)) + 'Hz.png')
                plt.close()

                # Crossing times and natural frequency
                fig4 = plt.figure(figsize=(10, 8))
                xyi4 = fig4.add_subplot(111)
                zerocrossings = np.where(np.diff(np.signbit(normcorrelationfunc)))[0]
                maxofcrossing = np.size((zerocrossings))
                numberofcrossings = np.linspace(1, maxofcrossing, maxofcrossing)
                xyi4.plot(time[1 + zerocrossings], numberofcrossings)
                pvar = np.polyfit(time[1 + zerocrossings], numberofcrossings, deg=1)
                dampedfreq = pvar[0] / 2
                print('dampedfreq', dampedfreq)
                xyi4.set_xlabel('Time (s)', fontsize=15)
                xyi4.set_ylabel('Zero crossing', fontsize=15)
                xyi4.set_title('Zero crossing for peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))),
                               fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                fig4.savefig(self.resultsfolder + self.filename[:-4] + names.ZEROCROSSING + '-' + str(
                    np.around(self.frequencyrange[peak], 2)) + 'Hz.png')
                plt.close()
                naturalfrequency = dampedfreq / np.sqrt(1 - np.square(dampingfactor))
                decayfactor = dampingfactor * naturalfrequency
                print('naturalfrequency', naturalfrequency)
                #print('decayfactor', decayfactor)
            else: #if there is less than 3 datapoints in the region
                print('Not enough data points in the ROI for frequency ' + str(np.around(self.frequencyrange[peak],2)))
                print('Setting damping factor to zero')
                print('Check bell shape curve and the test curve for the correlation function')

                figteste = plt.figure()
                xy = figteste.add_subplot(111)
                xy.scatter(correlationpeaks,normcorrelationfunc[correlationpeaks],marker='+')
                xy.plot(normcorrelationfunc)
                figteste.savefig(self.resultsfolder + 'test-curve-correlfunc-' + str(np.around(self.frequencyrange[
                                                                                                   peak],
                                                                                     2))+'.png')
                plt.close()
                naturalfrequency = self.frequencyrange[peak]
                dampingfactor = np.nan

            # Enhanced shape form (weighted sum)
            weightedshape = np.zeros((self.numofchannelsnew, self.numofchannelsnew))
            for svdline in range(self.numofchannelsnew):
                weightedshape[:, svdline] = self.Victorsingularvalues[peak, svdline] * self.left[peak, :, svdline]

            # newmodalshape = np.sum(weightedshape,axis=1)
            # newmodalshape = np.sum(self.left[peak,:,0])*newmodalshape/np.sum(newmodalshape)
            newmodalshape = self.left[peak, :, 0]
            # 2D Figure for mode shapes visualization
            figrawmodeshape = plt.figure(figsize=(15, 8))
            xyrawmodeshape = figrawmodeshape.add_subplot(111)
            xaxismodeshape = np.linspace(0, self.numofchannelsnew, self.numofchannelsnew, endpoint=False)
            xyrawmodeshape.plot(xaxismodeshape, newmodalshape, label='EFDD mode shape')
            xyrawmodeshape.plot(xaxismodeshape, self.left[peak, :, 0], label='FDD mode shape')
            xyrawmodeshape.set_xlabel('Channel', fontsize=15)
            xyrawmodeshape.set_ylabel('Amplitude (a.u.)', fontsize=15)
            xyrawmodeshape.set_title(
                'Mode shape for peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))), fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            figrawmodeshape.savefig(self.resultsfolder + self.filename[:-4] + names.EFDD_MODE_SHAPE + '-' + str(
                np.around(self.frequencyrange[peak], 2)) + 'Hz.png')
            plt.grid()
            plt.legend()
            # plt.show()
            plt.close()

            '''# 3D Figure for mode shapes visualization: works fine, but it doesnt bring much
            print('fig')
            figrawmodeshape = plt.figure(figsize=(15, 8))
            xyzrawmodeshape = figrawmodeshape.add_subplot(111, projection='3d')
            sensorposition = np.linspace(0,self.numofchannelsnew//3,self.numofchannelsnew//3, endpoint=False) #considering 3 axis sensors
            xmode = [newmodalshape[int(3*i)] for i in range(self.numofchannelsnew//3)]
            ymode = [newmodalshape[int(3 * i + 1)] for i in range(self.numofchannelsnew // 3)]
            zmode = [newmodalshape[int(3 * i + 2)] for i in range(self.numofchannelsnew // 3)]
            xyzrawmodeshape.plot(xmode,ymode,zs=zmode)
            xyzrawmodeshape.set_xlabel('X Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_ylabel('Y Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_ylabel('Z Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_title('Mode shape for peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak],2))), fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()
            pĺt.close()'''

            # 3D Animation for mode shapes visualization: video works but with 2 sensors it doesnt make sense, must test for 3 sensors
            figrawmodeshape = plt.figure(figsize=(15, 8))
            xyzrawmodeshape = figrawmodeshape.add_subplot(111, projection="3d")
            line, = xyzrawmodeshape.plot([], [], zs=[])
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
            sensorposition = np.linspace(0, self.numofchannelsnew // 3, self.numofchannelsnew // 3,
                                         endpoint=False)  # considering 3 axis sensors
            xyzrawmodeshape.set_xlabel('X Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_ylabel('Y Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_ylabel('Z Amplitude (a.u.)', fontsize=15)
            xyzrawmodeshape.set_title(
                'Mode shape for peak at {0} Hz'.format(str(np.around(self.frequencyrange[peak], 2))), fontsize=15)

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            sizeofsimulation = 625

            xmode, ymode, zmode = [np.zeros(int(self.numofchannelsnew // 3)) for i in range(3)]
            for dimension in range(int(self.numofchannelsnew // 3)):
                xmode[dimension] = np.array([newmodalshape[3 * dimension]])
                ymode[dimension] = np.array([newmodalshape[3 * dimension + 1]])
                zmode[dimension] = np.array([newmodalshape[3 * dimension + 2]])

            tdata = np.exp(25 * complex(0, 1) * np.linspace(0, 2 * np.pi, sizeofsimulation))

            vector = np.zeros((sizeofsimulation, int(self.numofchannelsnew // 3) - 1, 3))
            xmodetime, ymodetime, zmodetime = [np.zeros((sizeofsimulation, int(self.numofchannelsnew // 3))) for i in
                                               range(3)]
            # xmode[0], ymode[0], zmode[0] = [0, 0, 0]
            for dimension in range(int(self.numofchannelsnew // 3) - 1):
                xmodetime[:, dimension], ymodetime[:, dimension], zmodetime[:, dimension] = xmode[dimension] * tdata, \
                                                                                            ymode[dimension] * tdata, \
                                                                                            zmode[dimension] * tdata
                vector[:, dimension] = np.array([-xmodetime[:, dimension] + xmodetime[:, dimension + 1],
                                                 -ymodetime[:, dimension] + ymodetime[:, dimension + 1],
                                                 -zmodetime[:, dimension] + zmodetime[:, dimension + 1]]).transpose()
                vector[:, dimension] = vector[:, dimension] / np.linalg.norm(vector[:, dimension])

            dataLines = np.array([xmodetime, ymodetime, zmodetime])

            def init():
                line, = xyzrawmodeshape.plot([], [], zs=[])
                return xyzrawmodeshape,

            def animate(step):

                """xdata = dataLines[0,:]*tdata[25*step]
                ydata = dataLines[1,:]*tdata[25*step]
                zdata = dataLines[2,:]*tdata[25*step]"""
                plt.cla()
                xyzrawmodeshape.set_xlim(-np.amax(np.abs(xmode)), np.amax(np.abs(xmode)))
                xyzrawmodeshape.set_ylim(-np.amax(np.abs(ymode)), np.amax(np.abs(ymode)))
                xyzrawmodeshape.set_zlim(-np.amax(np.abs(zmode)), np.amax(np.abs(zmode)))
                xdata = dataLines[0, step]
                ydata = dataLines[1, step]
                zdata = dataLines[2, step]

                # localvector = np.zeros((int(self.numofchannelsnew // 3 - 1),3))
                # for dimension in range(self.numofchannelsnew // 3 - 1):
                # localvector[dimension] = np.array([-xdata[dimension] + xdata[dimension + 1], -ydata[dimension] + ydata[dimension + 1],-zdata[dimension] + zdata[dimension + 1]])
                # localvector[dimension] = localvector[dimension] / np.linalg.norm(localvector[dimension])
                # xdata[dimension + 1], ydata[dimension + 1], zdata[dimension + 1] = xdata[dimension] + localvector[dimension, 0], ydata[dimension] + localvector[dimension, 1], zdata[dimension] + localvector[dimension, 2]
                tex = ['Dish', 'CSS', 'Cam.']
                newxposition, newyposition, newzposition = [np.zeros((int(self.numofchannelsnew // 3))) for i in
                                                            range(3)]
                counter = 0
                for position in [2, 0, 1]:
                    newxposition[counter], newyposition[counter], newzposition[counter] = xdata[position], ydata[
                        position], zdata[position]
                    xyzrawmodeshape.text(newxposition[counter], newyposition[counter], newzposition[counter],
                                         tex[counter])
                    counter = counter + 1
                line, = xyzrawmodeshape.plot(newxposition, newyposition, zs=newzposition)
                xyzrawmodeshape.set_title('Mode shape at ' + str(np.around(self.frequencyrange[peak], 2)) + ' Hz',
                                          fontsize=15)
                xyzrawmodeshape.set_xlabel('X axis (a.u.)', fontsize=15)
                xyzrawmodeshape.set_ylabel('Y axis (a.u.)', fontsize=15)
                xyzrawmodeshape.set_zlabel('Z axis (a.u.)', fontsize=15)

                return xyzrawmodeshape,

            ani = animation.FuncAnimation(figrawmodeshape, animate, init_func=init, frames=25, repeat=False, blit=False)
            ani.save(self.resultsfolder + self.filename[:-4] + names.EFDD_MODE_SHAPE + '-' + str(
                np.around(self.frequencyrange[peak], 2)) + 'Hz.mp4', writer=writer)
            # plt.show()
            # plt.close()

            # Storing new variables in a matrix
            self.efdddampingratio.append(dampingfactor)
            self.efddmodalfreqs.append(naturalfrequency)
            self.enhancedmodalshape.append(newmodalshape)

        print(self.efdddampingratio)
        np.savetxt(self.resultsfolder + self.filename[:-4] + names.EFDD_FREQ_DAMP + '-' + str(
            np.around(self.freal, 1)) + names.DECIMATION + '.txt',
                   [self.efddmodalfreqs, self.efdddampingratio])

        np.savetxt(self.resultsfolder + self.filename[:-4] + names.EFDD_MODE_SHAPE + str(
            np.around(self.freal, 1)) + names.DECIMATION + '.txt',
                   self.enhancedmodalshape)
        plt.close()
        return self.efddmodalfreqs, self.efdddampingratio, self.enhancedmodalshape

    def spectrogram(self, channel):
        countingch = np.array(np.where(channel in self.sensors))[0, 0]
        f, t, Sxx = signal.spectrogram(self.Victordecimated[:, countingch], fs=self.desiredmaxfreq * 2)
        Sxx = np.log10(Sxx)
        fig = plt.figure(figsize=(10, 8))
        xyi = fig.add_subplot(111)
        # this index is between the selected channels already in self.sensors
        xyi.pcolormesh(t, f, Sxx)
        xyi.set_title('Spectrogram for Ch.' + str(channel), fontsize=15)
        xyi.set_ylabel('Frequency(Hz)', fontsize=15)
        xyi.set_xlabel('Time(s)', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        fig.savefig(self.resultsfolder + self.filename[:-4] + names.SPECTOGRAM + str(channel + 1) + '.png')
        plt.close()

        """def calibrate(self,**kwargs):
        self.voltagerange = kwargs.get('voltagerange', np.array([4,20])) # in mA (after convertion)
        self.gravityrange = kwargs.get('gravityrange', np.array([-2,2])) #in g units
        #self.selectchannels = kwargs.get('selectchannels', np.array(np.linspace(0,self.numofchannels,endpoint=False))) #in g units
        self.selectchannels = kwargs.get('selectchannels', self.sensors) #in g units

        self.Calibrateddata = np.zeros((np.size(self.Victorinput,axis=0),np.size(self.selectchannels)))
        for channelnum in [0, 1, 3, 4, 6, 7]:  # X and Y normally converted
            self.Calibrateddata[:, channelnum] = np.diff(self.gravityrange) * (
                        9.80665*self.Victorinput[:, self.correctedorder[channelnum]] - 12) / np.diff(self.voltagerange)
        for channelnum in [2, 5, 8]:  # Z must have 1g subtracted
            self.Calibrateddata[:, channelnum] = 9.80665*np.diff(self.gravityrange) * (self.Victorinput[:, self.correctedorder[channelnum]] - 8) / np.diff(
                self.voltagerange)
        # Because the sensor on the camera is 180 degree rotated from the other two sensors (invert Y and Z direction of the the other two sensors):
        for channelnum in [1, 2, 7, 8]:  # Y and Z must be inverted for CSS and dish
            self.Calibrateddata[:, channelnum] = -self.Calibrateddata[:, channelnum]

        return self.Calibrateddata"""

    def sensorshift(self, **kwargs):
        # Coordinate system: center of central plane is the origin
        # self.coordinates = kwargs.get('coordinates', np.array([[-2100,-2890,0],[1650,1650,4900],[3000,-5400,-10570]]))
        self.coordinates = kwargs.get('coordinates',
                                      np.array([[1.650, 1.650, 4.900], [3.000, -5.400, -10.570]]))
        numofsensors = self.numofchannelsnew // 3  # for tri axial sensors
        sizeofdata = np.size(self.Victorinput, axis=0)

        self.sensordisplacement = np.zeros((self.numofchannelsnew, sizeofdata - 2))

        # Calculating displacement for individual channels
        for sensors in range(self.numofchannelsnew):
            firstintegral = np.zeros(sizeofdata - 1)
            for step in range(sizeofdata - 1):
                firstintegral[step] = (self.Calibrateddata[step + 1, sensors] + self.Calibrateddata[step, sensors]) / (
                        self.samplingratio * 2)
            secondintegral = np.zeros((sizeofdata - 2))
            for step in range(sizeofdata - 2):
                secondintegral[step] = (firstintegral[step + 1] + firstintegral[step]) / (self.samplingratio * 2)
            self.sensordisplacement[sensors] = secondintegral

        # Comparing two channels to extract shift
        self.sensorshift = np.zeros((3, numofsensors, numofsensors, sizeofdata - 2))
        for sensors1 in range(numofsensors):
            for sensors2 in range(numofsensors):
                self.sensorshift[0, sensors1, sensors2] = (
                        self.sensordisplacement[sensors1 * 3] - self.sensordisplacement[
                    sensors2 * 3])  # /np.abs(self.coordinates[sensors2,0] - self.coordinates[sensors1,0])
                self.sensorshift[1, sensors1, sensors2] = (
                        self.sensordisplacement[sensors1 * 3 + 1] - self.sensordisplacement[
                    sensors2 * 3 + 1])  # /np.abs(self.coordinates[sensors2,1] - self.coordinates[sensors1,1])
                self.sensorshift[2, sensors1, sensors2] = (
                        self.sensordisplacement[sensors1 * 3 + 2] - self.sensordisplacement[
                    sensors2 * 3 + 2])  # /np.abs(self.coordinates[sensors2,2] - self.coordinates[sensors1,2])
                self.sensorshift[0, sensors1, sensors2] = self.sensorshift[0, sensors1, sensors2] - self.sensorshift[
                    0, sensors1, sensors2, 0]
                self.sensorshift[1, sensors1, sensors2] = self.sensorshift[1, sensors1, sensors2] - self.sensorshift[
                    1, sensors1, sensors2, 0]
                self.sensorshift[2, sensors1, sensors2] = self.sensorshift[2, sensors1, sensors2] - self.sensorshift[
                    2, sensors1, sensors2, 0]

        self.sensorshiftdecimated = signal.decimate(signal.decimate(self.sensorshift, 10, axis=-1), 10, axis=-1)
        newsize = np.size(self.sensorshiftdecimated, axis=-1)
        newtime = np.linspace(0, np.amax(self.time), newsize)

        dimlabel = ['X axis', 'Y axis', 'Z axis']
        for sensors1 in range(numofsensors):
            for sensors2 in range(numofsensors):
                if sensors2 > sensors1:
                    figshifts = plt.figure(figsize=(15, 8))
                    xyshifts = figshifts.add_subplot(111)
                    for dimension in range(3):
                        xyshifts.plot(newtime[1:], np.power(10, 6) * ((self.sensorshiftdecimated[
                            dimension, sensors1, sensors2]).transpose())[1:], label=str(dimlabel[dimension]))
                    xyshifts.set_title('Shift between Ch. ' + str(sensors1 + 1) + '-Ch. ' + str(sensors2 + 1),
                                       fontsize=15)
                    xyshifts.set_ylabel(r'Shift ($\mu$m)', fontsize=15)
                    xyshifts.set_xlabel('Time(s)', fontsize=15)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.legend()
                    figshifts.savefig(
                        self.resultsfolder + self.filename[:-4] + names.SHIFT + str(sensors1 + 1) + '-Ch. ' + str(
                            sensors2 + 1) + '.png')
                    np.savetxt(
                        self.resultsfolder + self.filename[:-4] + names.SHIFT + str(sensors1 + 1) + '-Ch. ' + str(
                            sensors2 + 1) + '.txt', self.sensorshiftdecimated[:, sensors1, sensors2])
                    plt.close()
        return self.sensorshiftdecimated


if __name__ == "__main__":
    teste = OMA('/scratch/users/vimartin/data/2019-09-30_15_files-merged.txt')
    teste.rawdataplot18()
    teste.calibrate()
    frequencyrange, yaxis, left, right, DONTRUNFLAG = teste.FDD(desiredmaxfreq=10, numoflinestoplot=3)
    if DONTRUNFLAG == True:
        print("OMA Analysis already ran for this file")
    else:
        peaks = teste.peakpicking()
        MACmatrix = teste.MACfunction()
        teste.MACplot(MACmatrix)
        newmodalfreq, goodindexinfullrange, newMACmatrix = teste.MACselection(MACmatrix)
        teste.MACplot(newMACmatrix)
        teste.enhancedfdd()