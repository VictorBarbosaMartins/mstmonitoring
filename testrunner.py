from Sample import onefileanalysis
from Sample import multiplefilesanalysis

"""#Test one file OMA
teste = onefileanalysis.OMA('08-07-2019-stabilitystudies-5files.txt')
teste.rawdataplot18()
frequencyrange, yaxis, left, right = teste.FDD(desiredmaxfreq=10, numoflinestoplot=1, sensors=[9,10,11,15,16,17])
teste.spectrogram(channel=9)
peaks = teste.peakpicking()
MACmatrix = teste.MACfunction()
teste.MACplot(MACmatrix)
newmodalfreq, goodindexinfullrange, newMACmatrix = teste.MACselection(MACmatrix)
teste.MACplot(newMACmatrix)
teste.enhancedfdd()
teste.calibrate()
teste.sensorshift()"""
'''
#Test one file weather
A = onefileanalysis.Weather('weatherData20190703.txt')
A.Analysis(timeofacquisition=(7,7.5))
A.Statistics()
A.Qualitycheck()
'''

#Test multiple file weather
#B= multiplefilesanalysis.Weather()
#Flags = B.Analysis(timeofacquisition=(7,7.5))
#Goodfiles = B.Selectionfiles()
#print(Goodfiles)

#Test multiple file OMA
#A = multiplefilesanalysis.RunOMA() #exclude files=Goodfiles to analyse all files
#A.sample()
B = multiplefilesanalysis.Track() #exclude files=Goodfiles to analyse all files
B.Trendanalysis()

