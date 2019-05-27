from Sample import onefileanalysis
teste = onefileanalysis.OMA('30-04-2019-sr100-10files-0elev.txt')
frequencyrange, yaxis, left, right = teste.FDD(desiredmaxfreq=15)
peaks = teste.peakpicking(frequencyrange,yaxis)
MACmatrix = teste.MAC(frequencyrange, peaks, left, right)
teste.MACplot(frequencyrange, peaks, MACmatrix)
newmodalfreq, goodindexinfullrange, newMACmatrix = teste.MACselection(frequencyrange, peaks, MACmatrix)
teste.MACplot(frequencyrange, goodindexinfullrange, newMACmatrix)
#consertar macplot