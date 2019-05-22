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
#This class analyses the data from the accelerometers, generates graphs and txt files with results

    def __init__(self, path, filename, samplingratio):
        self.filename = filename
        self.path = path
        self.samplingratio = samplingratio
        self.Victorinput = np.loadtxt(self.path + self.filename, delimiter=' ', dtype=float)

        # Number of sample points
        self.N = np.size(self.Victorinput, axis=0)
        self.dt = 1.0 / samplingratio

        # Number of valid channels
        self.numofchannels = np.size(self.Victorinput, axis=1) - 6  # because now we have 6 inclinometers

        # remove it once we have no inclinometers anymore

        # Nyquist frequency
        self.nyquist = samplingratio / 2

        # CREATE DIRECTORY TO STORE THE RESULTS
        try:
            os.makedirs(path + 'output')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def rawdataplot18(self):

        # Plot in a 6x3 window all the 18 channels
        filename = self.filename
        samplingratio = self.samplingratio
        numofchannels = self.numofchannels
        N = self.N
        dt = self.dt
        Victorinput = self.Victorinput
        x = np.fft.fftfreq(N, d=dt)
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
        f0.savefig(path + 'output\\' + filename[:-4] + '-rawdata-18.png')
        plt.close(f0)

    def rawdataplot(self, **kwargs):

        # Plot one channel
        self.channel = kwargs.get('channel', 7)
        filename = self.filename
        samplingratio = self.samplingratio
        numofchannels = self.numofchannels
        N = self.N
        dt = self.dt
        Victorinput = self.Victorinput
        x = np.fft.fftfreq(N, d=dt)
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
        f0.savefig(path + 'output\\' + filename[:-4] + '-rawdata-Ch.' + str(self.channel) + '.png')
        plt.close(f0)

    def FDD(self, desiredmaxfreq, **kwargs):

        self.desiredmaxfreq = desiredmaxfreq
        self.resolution = kwargs.get('resolution', 2048)
        self.datapoints = kwargs.get('datapoints', self.N)
        self.numoflinestoplot = kwargs.get('numoflinestoplot', 1)
        self.sensors = kwargs.get('sensors', np.arange(0, self.numofchannels))
        filename = self.filename
        numofchannelsnew = np.size(self.sensors)
        numberoffreqlines = np.round(self.resolution / 2).astype(int) + 1
        percentagetooverlap = 0.66
        factorofdecimation = int(self.nyquist / self.desiredmaxfreq - self.nyquist % self.desiredmaxfreq)
        outputresolution = self.N / factorofdecimation
        freal = self.nyquist / factorofdecimation  # shifted from desiredmaxfreq such as the factor of decimation become an integer
        sampleratingdecimated = 2 * freal
        timedecimated = np.linspace(0, self.N * self.dt, outputresolution)
        xyi = np.zeros((numofchannelsnew), dtype=object)
        Victordecimated = []
        Victorspectraldensity = np.zeros((numberoffreqlines, numofchannelsnew, numofchannelsnew))
        Victorsingularvalues = np.zeros((numberoffreqlines, numofchannelsnew))
        Victorsingularleftvector, Victorsingularrightvector = [
            np.zeros((numberoffreqlines, numofchannelsnew, numofchannelsnew)) for i in range(2)]

        for sensor in self.sensors:

            # SCALING/DETRENDING
            preprocessing.scale(self.Victorinput[:, sensor], with_mean=True, with_std=True, copy=False)

            # DECIMATING
            Victordecimated.append(signal.decimate(self.Victorinput[:self.N, sensor], factorofdecimation))
        Victordecimated = np.array(Victordecimated).transpose()

        # CROSS SPECTRAL DENSITY (CSD)
        for sensor1 in range(numofchannelsnew):
            for sensor2 in range(numofchannelsnew):
                frequencyrange, Victorspectraldensity[:, sensor1, sensor2] = signal.csd(Victordecimated[:, sensor1],
                                                                                        Victordecimated[:, sensor2],
                                                                                        fs=sampleratingdecimated,
                                                                                        nperseg=self.resolution,
                                                                                        noverlap=int(
                                                                                            percentagetooverlap * self.resolution),
                                                                                        detrend=False)
        # SVD
        reference = 400000  # 0.4 (km/s²)²
        for frequencyline in range(numberoffreqlines):
            Victorsingularleftvector[frequencyline], Victorsingularvalues[frequencyline], Victorsingularrightvector[
                frequencyline] = np.linalg.svd(
                Victorspectraldensity[frequencyline].reshape(1, numofchannelsnew, numofchannelsnew), full_matrices=True,
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
        plt.yticks(fontsize=20)
        plt.grid()
        plt.xlim(0, self.desiredmaxfreq)
        plt.ylim(-120, -50)
        # plt.ylim(-100,20)
        plt.title('OMA spectrum', fontsize=15)
        if numofchannelsnew == self.numofchannels:
            fig.savefig(path + 'output\\' + filename[:-4] + '-FDD-allsensors-' + str(freal) + 'Hzdec.png')
            np.savetxt(path + 'output\\' + filename[:-4] + '-FDD-allsensors-' + str(freal) + 'Hzdec-singvalues.txt',
                       Victorsingularvalues)
            np.savetxt(path + 'output\\' + filename[:-4] + '-FDD-allsensors-' + str(freal) + 'Hzdec-frequencies.txt',
                       frequencyrange)

        elif numofchannelsnew == 1:
            fig.savefig(path + 'output\\' + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' + str(
                freal) + 'Hzdec.png')
            np.savetxt(path + 'output\\' + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' + str(
                freal) + 'Hzdec-singvalues.txt', Victorsingularvalues)
            np.savetxt(path + 'output\\' + filename[:-4] + '-FDD-' + 'sensor-' + str(self.sensors) + '-' + str(
                freal) + 'Hzdec-frequencies.txt', frequencyrange)
        else:
            print('Fig. not saved - The figure is saved either for one or all sensors')
        plt.show()
        plt.clf()
        plt.close(fig)
        return frequencyrange, yaxis, Victorsingularleftvector, Victorsingularrightvector

    def peakpicking(self, frequencyrange, Victorsingularvalues, **kwargs):
        self.rangeofinterest = kwargs.get('roi', (0.5,
                                                  7.5))  # Freq. range to monitor the peaks, defined as the region where the modes tend to be easily excited (appear in every data taking)
        self.numoflinestoplot = kwargs.get('graphlines', 1)
        self.inputheight = kwargs.get('inputheight', 10)  # Minimum height of peak in dB in relation to ground
        self.distance = kwargs.get('distance', 5)  # Minimum distance between peaks
        self.width = kwargs.get('width', 3)  # Minimum width for the peak

        filename = self.filename
        size = np.size((Victorsingularvalues), axis=0)
        numofchannelsnew = np.size((Victorsingularvalues), axis=1)
        peaksstorage, peaksstorage2 = [], []
        numberoffreqlines = np.size(Victorsingularvalues, axis=0)

        for sensor in range(numofchannelsnew):
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
        numberofpeaks = np.zeros((numofchannelsnew), dtype=int)

        for singvalueline in range(self.numoflinestoplot):
            numberofpeaks[singvalueline] = np.size(
                peaksstorage[singvalueline])  # number of peaks in each singular value line
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
        f9.savefig(path + 'output\\' + filename[:-4] + '-FDD-peaks.png')
        plt.show()
        plt.clf()
        plt.close(f9)
        del f9
        return peaksstorage

    '''def trackmodes(self,frequencyrange,peaksstorage):
        peaks = peaksstorage[0]
        f10 = plt.figure(figsize=(10,8))
        xyi = f10.add_subplot(111)'''

    # MODAL ASSURANCE CRITERION (MAC) - FIRST CALCULATION
    def MAC(self, frequencyrange, peaksstorage, left, right):
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
        # PLOT
        peakindexes = peakindexes.astype(int)
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
        f10.savefig(path + 'output\\' + self.filename[:-4] + '-MACvalues-allpeaks.png')
        plt.show()
        plt.clf()
        plt.close(f10)
        del f10

    def MACselection(self, frequencyrange, peaksstorage, MAC, **kwargs):
        # Delete linearly dependent potential modal frequencies
        # Preserving the pot. modal freq. with smaller Sum of MAC
        # If the Sum of MACs are equal, preserve the lower freq
        self.maclimit = kwargs.get('maclimit', 0.15)
        numofpeaks = np.size(peaksstorage[0])
        includedindexes = np.where(MAC > self.maclimit)  # and !=1
        modalfreq = np.array([frequencyrange[peaksstorage[0]]])[0, :]
        # print('modalfreq',modalfreq)
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
        print(goodindexinfullrange)
        print(goodindexinmodalfreq)
        for newpeak1 in range(newnumberofpeaks):
            for newpeak2 in range(newnumberofpeaks):
                newMACmatrix[newpeak1, newpeak2] = MAC[goodindexinmodalfreq[newpeak1], goodindexinmodalfreq[newpeak2]]
        return newmodalfreq, goodindexinfullrange, newMACmatrix
