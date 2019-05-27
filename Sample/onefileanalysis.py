import numpy as np
import scipy
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from glob import glob
from scipy import signal, interpolate
from sklearn import preprocessing
from scipy.stats import chisquare
import os, errno
import matplotlib as m


class OMA(object):
    '''This class analyses the datafile from the accelerometers, generates graphs and txt files with results

    #Suggested name for the datafiles (supposed to be stored at ./data/:
    #YYYY-MM-DD-MST-XX-STR.txt, where XX indicates the telescope

    Default Sampling ratio is 100 Hz;'''

    def __init__(self, filename):
        self.filename = filename
        self.path = os.getcwd()
        self.samplingratio = 100 # Samp. ratio = 100 Hz. You can change it, but it should be a defined value forever
        self.Victorinput = np.loadtxt(self.path + '/data/' + self.filename, delimiter=' ', dtype=float)

        # Number of sample points
        self.N = np.size(self.Victorinput, axis=0)
        self.dt = 1/self.samplingratio

        # Number of valid channels
        self.numofchannels = np.size(self.Victorinput, axis=1) - 6  # because now we have 6 inclinometers
        # remove it once we have no inclinometers anymore

        # Nyquist frequency
        self.nyquist = self.samplingratio/2

        # Create directories to store results
        self.rawdatafolder = '/output/onefileanalysis/rawdata/'
        self.resultsfolder = '/output/onefileanalysis/analysisresults/'
        createdirectories = [self.rawdatafolder,self.resultsfolder]
        for directory in createdirectories:
            try:
                os.makedirs(self.path + directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def rawdataplot18(self):
        '''Plot 18 channels in one window.
        to do: Remove/alter function once the number of available channels changes'''
        filename = self.filename
        samplingratio = self.samplingratio
        numofchannels = self.numofchannels
        N = self.N
        dt = self.dt
        Victorinput = self.Victorinput
        time = np.linspace(0, N * dt, N)
        f0 = plt.figure(figsize=(15, 8))
        xyi = np.zeros((numofchannels), dtype=object)
        for sensor in range(numofchannels):
            xyi[sensor] = f0.add_subplot(6, 3, sensor + 1)
            xyi[sensor].plot(time, Victorinput[:, sensor], linewidth=0.5)
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
        xyi[1].set_title('Raw data', fontsize=15)
        xyi[6].set_ylabel('Acceleration (mA)', fontsize=15)
        xyi[16].set_xlabel('Time(s)', fontsize=15)
        f0.savefig(path + self.rawdatafolder + filename[:-4] + '-raw-18.png')
        plt.close(f0)

    def rawdataplot(self, channel):
        '''Plot the raw data for one channel from 1 to number max. of channels'''
        filename = self.filename
        rawdatafolder = self.rawdatafolder
        N = self.N
        dt = self.dt
        Victorinput = self.Victorinput
        time = np.linspace(0, N * dt, N)
        f0 = plt.figure(figsize=(15, 8))
        xyi = f0.add_subplot(111)
        xyi.plot(time, Victorinput[:, self.channel - 1], linewidth=0.5)
        xyi.set_title('Raw data', fontsize=15)
        xyi.set_ylabel('Acceleration (mA)', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        xyi.set_xlabel('Time(s)', fontsize=15)
        f0.savefig(path + rawdatafolder + filename[:-4] + '-raw-Ch.' + str(self.channel) + '.png')
        plt.close(f0)

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

        self.desiredmaxfreq = kwargs.get('desiredmaxfreq', 10)
        self.resolution = kwargs.get('resolution', 2048)
        self.datapoints = kwargs.get('datapoints', self.N)
        self.numoflinestoplot = kwargs.get('numoflinestoplot', 1)
        self.sensors = kwargs.get('sensors', np.arange(0, self.numofchannels))
        resultsfolder = self.resultsfolder
        path = self.path
        filename = self.filename
        self.numofchannelsnew = np.size(self.sensors)
        numberoffreqlines = np.round(self.resolution / 2).astype(int) + 1
        percentagetooverlap = 0.66
        #factorofdecimation = int(self.nyquist / self.desiredmaxfreq - self.nyquist % self.desiredmaxfreq)
        factorofdecimation = int(np.around(self.nyquist / self.desiredmaxfreq))
        outputresolution = self.N / factorofdecimation
        self.freal = np.around (self.nyquist / factorofdecimation,1)  # shifted from desiredmaxfreq such as the factor of decimation become an integer
        sampleratingdecimated = 2 * self.freal
        timedecimated = np.linspace(0, self.N * self.dt, outputresolution)
        xyi = np.zeros((self.numofchannelsnew), dtype=object)
        Victordecimated = []
        Victorspectraldensity = np.zeros((numberoffreqlines, self.numofchannelsnew, self.numofchannelsnew))
        Victorsingularvalues = np.zeros((numberoffreqlines, self.numofchannelsnew))
        Victorsingularleftvector, Victorsingularrightvector = [
            np.zeros((numberoffreqlines, self.numofchannelsnew, self.numofchannelsnew)) for i in range(2)]

        for sensor in self.sensors:

            # SCALING/DETRENDING
            preprocessing.scale(self.Victorinput[:, sensor], with_mean=True, with_std=True, copy=False)

            # DECIMATING
            Victordecimated.append(signal.decimate(self.Victorinput[:self.N, sensor], factorofdecimation))
        Victordecimated = np.array(Victordecimated).transpose()

        # CROSS SPECTRAL DENSITY (CSD)
        for sensor1 in range(self.numofchannelsnew):
            for sensor2 in range(self.numofchannelsnew):
                frequencyrange, Victorspectraldensity[:, sensor1, sensor2] = signal.csd(Victordecimated[:, sensor1],
                                                                                        Victordecimated[:, sensor2],
                                                                                        fs=sampleratingdecimated,
                                                                                        nperseg=self.resolution,
                                                                                        noverlap=int(
                                                                                            percentagetooverlap * self.resolution),
                                                                                        detrend=False)
        # SVD
        reference = 400000  # 0.4 (km/s²)² was chosen to match the Artemis Modal software results
                            # This value may be changed but keep the same forever!
        for frequencyline in range(numberoffreqlines):
            Victorsingularleftvector[frequencyline], Victorsingularvalues[frequencyline], Victorsingularrightvector[
                frequencyline] = np.linalg.svd(
                Victorspectraldensity[frequencyline].reshape(1, self.numofchannelsnew, self.numofchannelsnew), full_matrices=True,
                compute_uv=True)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        xyi = fig.add_subplot(111)
        yaxis = np.zeros((numberoffreqlines, self.numoflinestoplot))
        for sensor in range(self.numoflinestoplot):
            yaxis[:, sensor] = 10 * np.log10(Victorsingularvalues[:, sensor] / reference)
            xyi.plot(frequencyrange, yaxis[:, sensor], linewidth=1)
            # xyi.plot(frequencyrange,Victorsingularvalues[:,sensor],linewidth=1)
            # xyi.set_yscale('log')
        plt.xlabel('Frequency (Hz)', fontsize=15)
        # plt.ylabel('Intensity('+r'$(m/s^2)²/Hz)$'+')',fontsize=15)
        plt.ylabel('Intensity(dB)', fontsize=20)
        plt.xticks(np.arange(min(frequencyrange), max(frequencyrange) + 1, 1), fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.xlim(0, self.desiredmaxfreq)
        plt.ylim(-120, -50)
        # plt.ylim(-100,20)
        plt.title('OMA spectrum', fontsize=15)
        if self.numofchannelsnew == self.numofchannels:
            fig.savefig(path + resultsfolder + filename[:-4] + '-FDD-allsensors-' + str(np.around(self.freal,1)) + 'Hzdec.png')
            np.savetxt(path + resultsfolder + filename[:-4] + '-FDD-allsensors-' + str(np.around(self.freal,1)) + 'Hzdec-singvalues.txt',
                       Victorsingularvalues)
            np.savetxt(path + resultsfolder + filename[:-4] + '-FDD-allsensors-' + str(np.around(self.freal,1)) + 'Hzdec-frequencies.txt',
                       frequencyrange)

        elif self.numofchannelsnew == 1:
            fig.savefig(path + resultsfolder + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' +
                str(np.around(self.freal,1)) + 'Hzdec.png')
            np.savetxt(path + resultsfolder + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' +
                str(np.around(self.freal,1)) + 'Hzdec-singvalues.txt', Victorsingularvalues)
            np.savetxt(path + resultsfolder + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' +
                str(np.around(self.freal,1)) + 'Hzdec-frequencies.txt', frequencyrange)
        else:
            print('Fig. not saved - The figure is saved either for only one sensor or all of them')
        plt.show()
        #plt.clf()
        plt.close(fig)
        return frequencyrange, yaxis, Victorsingularleftvector, Victorsingularrightvector

    def peakpicking(self, frequencyrange, Victorsingularvalues, **kwargs):

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

        self.rangeofinterest = kwargs.get('roi', (0.5,7.5))  # Freq. range to monitor the peaks, defined as the region where the modes tend to be easily excited (appear in every data taking)
        self.numoflinestoplot = kwargs.get('graphlines', 1)
        self.inputheight = kwargs.get('inputheight', 10)  # Minimum height of peak in dB in relation to ground
        self.distance = kwargs.get('distance', 5)  # Minimum distance between peaks
        self.width = kwargs.get('width', 3)  # Minimum width for the peak

        path = self.path
        filename = self.filename
        size = np.size((Victorsingularvalues), axis=0)
        numofchannelsforpeaks = np.size((Victorsingularvalues), axis=1)
        peaksstorage, peaksstorage2 = [], []
        numberoffreqlines = np.size(Victorsingularvalues, axis=0)

        for sensor in range(numofchannelsforpeaks):
            minindex = np.where(frequencyrange > self.rangeofinterest[0])[0][0]
            maxindex = np.where(frequencyrange < self.rangeofinterest[1])[0][-1]
            mininregionofinterest = np.amin(Victorsingularvalues[minindex:maxindex, sensor])
            maxinregionofinterest = np.amax(Victorsingularvalues[minindex:maxindex, sensor])
            height = mininregionofinterest + self.inputheight
            peaks, _ = signal.find_peaks(Victorsingularvalues[:, sensor], height=height, width=self.width,
                                         distance=self.distance)
            peaksstorage.append(peaks)
        peaksstorage = np.array(peaksstorage)

        # PLOT PEAKS
        f9 = plt.figure(figsize=(12, 8))
        xyi = f9.add_subplot(111)
        numberofpeaks = np.zeros((numofchannelsforpeaks), dtype=int)

        for singvalueline in range(self.numoflinestoplot):
            numberofpeaks[singvalueline] = np.size(peaksstorage[singvalueline])  # number of peaks in each singular value line
            xyi.plot(frequencyrange, Victorsingularvalues[:, singvalueline], linewidth=1)
            # xyi.plot(frequencyrange,np.ones(size)*averageofinterest,linewidth=1, c='black', label='Mean {0} - {1} Hz'.format(self.rangeofinterest[0],self.rangeofinterest[1]))
            # xyi.set_yscale('log')
            xyi.set_xlim(0, 10)
            # xyi.fill_betweenx(Victorsingularvalues[:,sensor], frequencyrange[minindex], frequencyrange[maxindex],
            # facecolor='green', alpha=0.3)
            # indexfreqofinterest = np.linspace(minindex,maxindex, dtype=int)
            xyi.scatter(frequencyrange[peaksstorage[singvalueline]],
                        Victorsingularvalues[peaksstorage[singvalueline], singvalueline], marker='+', color='red',
                        label='Potential modal frequencies')
        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.ylim(-110, -50)
        # plt.ylabel('Intensity('+r'$(m^2/s)/Hz)$'+')', fontsize=15)
        plt.ylabel('Intensity(dB)', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(np.arange(min(frequencyrange), max(frequencyrange) + 1, 1), fontsize=15)
        plt.grid()
        plt.title('OMA with frequency peaks', fontsize=15)
        plt.legend(fontsize=15, loc=3)
        plt.show()
        if self.numofchannelsnew == self.numofchannels:
            f9.savefig(path + self.resultsfolder + filename[:-4] + '-allsensors-' + str(
            np.around(self.freal,1)) + 'Hzdec-frequencies.txt ' + '-FDD-peaks.png')
        elif self.numofchannelsnew == 1:
            f9.savefig(path + self.resultsfolder + filename[:-4] + 'sensor-' + str(self.sensors) + '-' + str(
            np.around(self.freal,1)) + 'Hzdec-frequencies.txt ' + '-FDD-peaks.png')
        plt.close(f9)
        del f9
        return peaksstorage

    '''def trackmodes(self,frequencyrange,peaksstorage):
        peaks = peaksstorage[0]
        f10 = plt.figure(figsize=(10,8))
        xyi = f10.add_subplot(111)'''

    # MODAL ASSURANCE CRITERION (MAC) - FIRST CALCULATION
    def MAC(self, frequencyrange, peaksstorage, left, right):

        '''
        Calculate the MAC value for each possible pair of peak encountered in the peakpicking function.
        :param frequencyrange: Range of frequency (1-D numpy array);
        :param peaksstorage: Matrix which contains the position of the peaks in the frequencyrange;
        :param left: left singular vector resulted from the FDD method;
        :param right:  right singular vector resulted from the FDD method;
        :return: MACmatrix, matrix n x n with the MAC value for each pair of n encountered peaks.
        '''

        numberofpeaks = np.size(peaksstorage[0])  # number of peaks in each singular value line
        MACmatrix = np.zeros((numberofpeaks, numberofpeaks))  # MAC matrix for each singular value line
        peaknumber1 = 0
        for peak1 in peaksstorage[0]:
            peaknumber2 = 0
            for peak2 in peaksstorage[0]:
                MACmatrix[peaknumber1, peaknumber2] = np.square(np.dot(left[peak1, :, 0], left[peak2, :, 0])) / (
                            np.dot(left[peak1, :, 0], left[peak1, :, 0]) * np.dot(left[peak2, :, 0], left[peak2, :, 0]))
                peaknumber2 += 1
            peaknumber1 += 1
        return MACmatrix

    def MACplot(self, frequencyrange, peakindexes, MAC):

        '''
        Plot the MAC value for the inputed peaks
        :param frequencyrange: Range of frequency (1-D numpy array)
        :param peakindexes: n-D numpy array which contains the position of the peaks in the frequencyrange for the singular value lines
        :param MAC: n-D numpy array which contains the MAC-values for the n x n combination of peaks
        :return: plot
        '''

        peakindexes = peakindexes[0].astype(int)
        numberofpeaks = np.size(MAC, axis=0)
        f10 = plt.figure(figsize=(10, 8))
        xyi = f10.add_subplot(111)
        imagem = xyi.imshow(np.flip(MAC, axis=0), shape=(numberofpeaks, numberofpeaks), vmin=0, vmax=1)
        plt.locator_params(axis='both', nbins=numberofpeaks)
        labelinfreq = np.round(frequencyrange[peakindexes], 2).flatten()
        labelsx = [item.get_text() for item in xyi.get_xticklabels()]
        labelsy = [item.get_text() for item in xyi.get_yticklabels()]

        labelsx[1:] = labelinfreq
        labelsy[1:] = labelinfreq[::-1]
        xyi.set_xticklabels(labelsx, fontsize=15)
        xyi.set_yticklabels(labelsy, fontsize=15)

        plt.xticks(rotation='vertical')
        cbar = f10.colorbar(imagem)
        minvalue = np.amin(frequencyrange[peakindexes])
        maxvalue = np.amax(frequencyrange[peakindexes])
        xyi.set_xlabel('Freq.(Hz)', fontsize=15)
        xyi.set_ylabel('Freq.(Hz)', fontsize=15)
        xyi.set_title('MAC - Peaks', fontsize=15)
        f10.savefig(self.path + self.resultsfolder + self.filename[:-4] + '-MACvalues-allpeaks.png')
        plt.show()
        plt.close(f10)
        del f10
        return 0

    def MACselection(self, frequencyrange, peaksstorage, MAC, **kwargs):

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
        self.maclimit = kwargs.get('maclimit', 0.15)
        numofpeaks = np.size(peaksstorage[0])
        includedindexes = np.where(MAC > self.maclimit)  # and !=1
        modalfreq = np.array([frequencyrange[peaksstorage[0]]])[0, :]
        storeindextodelete = []
        badcorrelation1 = includedindexes[0]
        badcorrelation2 = includedindexes[1]
        for element in range(np.size(includedindexes[0])):
            if badcorrelation1[element] != badcorrelation2[element]:
                sumofmacs = np.array(
                    [np.sum(MAC[badcorrelation1[element], :]), np.sum(MAC[badcorrelation2[element], :])])
                indexofmaxMAC = np.argmax(sumofmacs)
                if sumofmacs[0] == sumofmacs[1]:
                    deleteindex = np.amax([badcorrelation1[element], badcorrelation2[element]])
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
            goodindexinfullrange[newpeak1] = np.array(np.where(frequencyrange == newmodalfreq[newpeak1])).flatten()
            goodindexinmodalfreq[newpeak1] = np.array(np.where(modalfreq == newmodalfreq[newpeak1])).flatten()
        for newpeak1 in range(newnumberofpeaks):
            for newpeak2 in range(newnumberofpeaks):
                newMACmatrix[newpeak1, newpeak2] = MAC[goodindexinmodalfreq[newpeak1], goodindexinmodalfreq[newpeak2]]
        return newmodalfreq, goodindexinfullrange, newMACmatrix
