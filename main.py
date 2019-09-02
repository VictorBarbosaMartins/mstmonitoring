from Sample import weathermultiple, weathersingle, omamultiple, omasingle

# Test one file OMA
# teste = onefileanalysis.OMA('structure_daq1.6.100_telescopeMST0__0_2019-08-25_05-33-33_960000.ascii_2019-08
# -25_15_files-merged.txt')



#Test one file weather
A = weathersingle.Weather('weatherData20190703.txt')
help(A.Analysis)
A.Analysis(timeofacquisition=(7,7.5))
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


"""#ALL TOGETHER
#A = convertfile.Manipulatedata()
#A.convertallascii(searchstring='*2019-08-2*_0*')
#A.mergefilesall(rangeoffiles=[0,15], dates=['2019-08-20','2019-08-28'])
#C= multiplefilesanalysis.Weather()
#Flags = C.Analysis(files=Selectedfiles,timeofacquisition=(6,7))
#Goodfiles = C.Selectioncriteria()
#Selectedfiles = C.Selectionfiles(dates=['2019-08-16','2019-08-25'])
#A = multiplefilesanalysis.RunOMA(files=Selectedfiles)
#A.sample()
B = multiplefilesanalysis.Track()
B.Trendanalysis(numofchannels=9)
R = reports.Reports()
R.generate()
R.sendemail()"""
