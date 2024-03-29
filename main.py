from datetime import datetime
from datetime import timedelta
from Sample import weathermultiple, weathersingle, omamultiple, omasingle, convertfile, reports
import definitions, argparse


###### TESTS ######

# Test one file OMA
# teste = onefileanalysis.OMA('structure_daq1.6.100_telescopeMST0__0_2019-08-25_05-33-33_960000.ascii_2019-08
# -25_15_files-merged.txt')


#Test one file weather
#A = weathersingle.Weather('weatherData20190703.txt')
#help(A.Analysis)
#A.Analysis(timeofacquisition=(7,7.5))
#A.Statistics()
#A.Qualitycheck()


# Test multiple file OMA and Weather
# C= multiplefilesanalysis.Weather()
# Selectedfiles = C.Selectionfiles(dates=['2019-08-16','2019-08-23'])
# Flags = C.Analysis(files=Selectedfiles,timeofacquisition=(6,7))
# Goodfiles = C.Selectioncriteria()
# A = multiplefilesanalysis.RunOMA(files=Selectedfiles) #exclude files=Goodfiles to analyse all files
# A.sample()
# Flags = C.Analysis(timeofacquisition=(6,7), files=Selectedfiles)
# Goodfiles = C.Selectioncriteria()
# B = multiplefilesanalysis.Track() #exclude files=Goodfiles to analyse all files
# B.Trendanalysis(numofchannels=9
# R = reports.Reports()
# R.generate()
# R.sendemail()

# Test one convert file
# A = convertfile.Manipulatedata()
# A.convertascii(IN=os.getcwd()+'/data/structure_daq1.6.200_telescopeMST0__1_2019-08-15_09-09-15_830000.dat',OUT=os.getcwd()+'/data/structure_daq1.6.200_telescopeMST0__1_2019-08-15_09-09-15_830000.ascii')
# A.converttxt(IN=os.getcwd()+'/data/structure_daq1.6.100_telescopeMST0__0_2019-08-15_10-34-15_840000.ascii', OUT=os.getcwd() + '/data/structure_daq1.6.100_telescopeMST0__0_2019-08-15_10-34-15_840000.txt')

# Test all files convert
# A = convertfile.Manipulatedata()
# A.convertallascii(searchstring='*2019-08-2*_0*')

# A.mergefiles(date='2019-08-20',rangeoffiles=[45,50])
# A.mergefilesall(rangeoffiles=[0,15], dates=['2019-08-20','2019-08-23']) #excluding first dataset

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--versionversion'
                        #help='')
    #args = parser.parse_args()

    #ALL TOGETHER
    definitions.define_paths()
    todayis = datetime.now()
    todayisstring = str(todayis)[:10]
    onemonthbefore = str(todayis - timedelta(days=30))[:10]
    print('Today is', todayis)

    Manipulatingdata = convertfile.Manipulatedata()
    Manipulatingdata.convertallascii(searchstring='*1.6.100*')
    # The first 15 datasets must be the ones taken for the health monitoring system
    Manipulatingdata.mergefilesall(rangeoffiles=[0,15], dates=['2019-08-16','2020-02-01'])

    Weatheranalysis=weathermultiple.Weather()
    Selectedfiles = Weatheranalysis.Selectionfiles(dates=['2019-08-16','2020-02-01'])
    Flags = Weatheranalysis.Analysis(files=Selectedfiles,timeofacquisition=(6,7)) #uncomment if using selection criteria
    Goodfiles = Weatheranalysis.Selectioncriteria()
    Weatheranalysis.Windeveryday()
    Weatheranalysis.Tempeveryday()

    Runningoma = omamultiple.RunOMA(files=Goodfiles)
    Runningoma.sample()
    Tracking = omamultiple.Track(files=Selectedfiles)
    Tracking = omamultiple.Track(files=Goodfiles)
    Tracking.Trendanalysis(numofchannels=9)
    Summary = reports.Reports()
    Summary.generate()
    Summary.sendemail()




"""from datetime import datetime
from datetime import timedelta
from Sample import weathermultiple, weathersingle, omamultiple, omasingle, convertfile, reports
import numpy as np
import definitions
import matplotlib.style as style
style.use('seaborn-colorblind')
style.use('/afs/ifh.de/group/hess/scratch/user/vimartin/HESS/usefullscripts/morphology/matplot_default')
from matplotlib import pyplot as plt
plt.set_cmap("afmhot")

# Test one file OMA
# teste = onefileanalysis.OMA('structure_daq1.6.100_telescopeMST0__0_2019-08-25_05-33-33_960000.ascii_2019-08
# -25_15_files-merged.txt')



#Test one file weather
#A = weathersingle.Weather('weatherData20190703.txt')
#help(A.Analysis)
#A.Analysis(timeofacquisition=(7,7.5))
#A.Statistics()
#A.Qualitycheck()


# Test multiple file OMA and Weather

# C= multiplefilesanalysis.Weather()
# Selectedfiles = C.Selectionfiles(dates=['2019-08-16','2019-08-23'])
# Flags = C.Analysis(files=Selectedfiles,timeofacquisition=(6,7))
# Goodfiles = C.Selectioncriteria()
# A = multiplefilesanalysis.RunOMA(files=Selectedfiles) #exclude files=Goodfiles to analyse all files
# A.sample()

# Flags = C.Analysis(timeofacquisition=(6,7), files=Selectedfiles)
# Goodfiles = C.Selectioncriteria()
# B = multiplefilesanalysis.Track() #exclude files=Goodfiles to analyse all files
# B.Trendanalysis(numofchannels=9)

# R = reports.Reports()
# R.generate()
# R.sendemail()

# Test one convert file
# A = convertfile.Manipulatedata()
# A.convertascii(IN=os.getcwd()+'/data/structure_daq1.6.200_telescopeMST0__1_2019-08-15_09-09-15_830000.dat',OUT=os.getcwd()+'/data/structure_daq1.6.200_telescopeMST0__1_2019-08-15_09-09-15_830000.ascii')
# A.converttxt(IN=os.getcwd()+'/data/structure_daq1.6.100_telescopeMST0__0_2019-08-15_10-34-15_840000.ascii', OUT=os.getcwd() + '/data/structure_daq1.6.100_telescopeMST0__0_2019-08-15_10-34-15_840000.txt')

# Test all files convert
# A = convertfile.Manipulatedata()
# A.convertallascii(searchstring='*2019-08-2*_0*')

# A.mergefiles(date='2019-08-20',rangeoffiles=[45,50])
# A.mergefilesall(rangeoffiles=[0,15], dates=['2019-08-20','2019-08-23']) #excluding first dataset


#ALL TOGETHER
definitions.definepaths()
#todayis = datetime.now()
#todayisstring = str(todayis)[:10]
#onemonthbefore = str(todayis - timedelta(days=30))[:10]
#print('Today is', todayis)
#Manipulatingdata = convertfile.Manipulatedata()
#Manipulatingdata.convertallascii(searchstring='*1.6.100*20*')
# The first 15 datasets must be the ones taken for the health monitoring system
#Manipulatingdata.mergefilesall(rangeoffiles=[0,15], dates=['2019-08-16',todayisstring])

Weatheranalysis=weathermultiple.Weather()
#Selectedfiles = Weatheranalysis.Selectionfiles(dates=['2019-08-16',todayisstring])
Selectedfiles = Weatheranalysis.Selectionfiles(dates=['2019-11-07','2019-12-28'])
#Selectedfiles = Weatheranalysis.Selectionfiles(dates=['2019-12-27','2020-02-01'])

#for element in Selectedfiles:
    #telescope is moving during the time
    #if element.find('20191127') != -1:
        #Selectedfiles = np.delete(Selectedfiles,np.where(Selectedfiles==element))
Flags = Weatheranalysis.Analysis(files=Selectedfiles,timeofacquisition=(6,7)) #uncomment if using selection criteria
Goodfiles = Weatheranalysis.Selectioncriteria()

#for element in Goodfiles:
    #telescope is moving during the time
    #if element.find('20191127') != -1:
        #Goodfiles = np.delete(Goodfiles,np.where(Goodfiles==element))

Weatheranalysis.Windeveryday()
#Weatheranalysis.Tempeveryday()

#Use following line if using the selection criteria
##Runningoma = omamultiple.RunOMA(files=Goodfiles)
#Use following line if not using selection criteria
#Runningoma = omamultiple.RunOMA(files=Selectedfiles)

##Runningoma.sample()

#Use the following line if using the selection criteria
##Tracking = omamultiple.Track(files=Goodfiles)
#Use following line if not using the selection criteria
#Tracking = omamultiple.Track(files=Selectedfiles)

##Tracking.Trendanalysis(numofchannels=9)
#Summary = reports.Reports()
#Summary.generate()
#Summary.sendemail()"""