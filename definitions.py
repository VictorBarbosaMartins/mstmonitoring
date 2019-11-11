import os

def define_paths():
    # SETTING UP ENVIRONMENT VARIABLES
    os.environ["MST-STR-MON-HOME"] = '/home/victor/Desktop/teste'
    os.environ["MST-STR-MON-DATA"] = os.environ["MST-STR-MON-HOME"] + '/data'
    os.environ["MST-STR-MON-WEATHERDATA"] = os.environ["MST-STR-MON-DATA"] + '/weather'
    os.environ["MST-STR-MON-RESULTS"] = os.environ["MST-STR-MON-HOME"] + '/output'
    os.environ["MST-STR-MON-ONEFILERESULTS"] = os.environ["MST-STR-MON-RESULTS"] + '/onefileanalysis'
    os.environ["MST-STR-MON-RAWDATAPLOTS"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/rawdata'
    os.environ["MST-STR-MON-ONEFILEOMA"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/analysisresults'
    os.environ["MST-STR-MON-WEATHERRESULTS"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/weather'
    os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] = os.environ["MST-STR-MON-RESULTS"] + '/multiplefileanalysis'
    os.environ["MST-STR-MON-MULTIPLEFILESWEATHERRESULTS"] = os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] + '/weather'
    os.environ["MST-STR-MON-REPORTS"] = os.environ["MST-STR-MON-RESULTS"] + '/reports'
    os.environ["MST-STR-MON-DATA-CONVERTED"] = os.environ["MST-STR-MON-DATA"]

# SETTING UP NAMES OF FILES
MERGING = '_files-merged'
RAW_ALL = '_raw-data-all-Ch'
RAW_CH = '_raw-data-Ch'
WIND = '_wind'
TEMPERATURE = '_temp'
PRESSURE = '_pressure'
HUMIDITY = '_humidity'
RAINRATE = '_rain-rate'
FDD = '_FDD-Chs-'
PEAKS = '_peaks'
DECIMATION = '_dec'
CHANNELS_USED = '_Ch.'
MACVALUE_ALL = '_MAC-all'
MACVALUE_SELECTED = '_MAC-selected'
SPECTOGRAM = '_spect-Ch-'
EFDD_FREQ_DAMP = '_EFDD-modal-freqs'
EFDD_MODE_SHAPE = '_EFDD-mode-shapes'
SHIFT = '_shift-Ch-'
SHIFTAVG = '_shift-avg'
LOGDEC = '_log-dec'
ZEROCROSSING = '_zero-cross'
AUTOCORREL = '_auto-correl'
BELLSHAPE = '_bell-shape'
CHISQUARE_FREQ = '_chi-square-freq'
CHISQUARE_MODE_SHAPE = '_chi-square-mode-shape'
CHISQUARE_DAMPING = '_chi-square-damp'
ALLFILES = '_all-files'
SELECTEDFILES = '_selected-files'
CHANGE_FACTOR_FREQ = '_change-factor-freq'
CHANGE_FACTOR_DAMP = '_change-factor-damp'
NUM_CORRELATION = '_num-of-correl'
TRACK_FREQ = '_track-freq'
TRACK_MODE_SHAPE = '_track-mode-shape'
TRACK_DAMP = '_track-damp'
FREQXDAMP = '_freqxdamp'
REPORTS = 'Reports-'
EMAILS_LIST = 'emails-list'
WIND_DAYS = 'wind-days'


#TO USE IN THE LOCAL COMPUTER ONLY
"""import os

def definepaths():
# SETTING UP ENVIRONMENT VARIABLES
    os.environ["MST-STR-MON-HOME"] = '/home/vimartin'
    os.environ["MST-STR-MON-DATA"] = '/scratch/hadoop/gantner/gerrit'
    os.environ["MST-STR-MON-DATA-CONVERTED"] = '/scratch/users/vimartin/data'
    os.environ["MST-STR-MON-WEATHERDATA"] = '/home/actlop/WeatherStation/data'
    os.environ["MST-STR-MON-RESULTS"] = '/scratch/users/vimartin/output'
    os.environ["MST-STR-MON-ONEFILERESULTS"] = os.environ["MST-STR-MON-RESULTS"] + '/onefileanalysis'
    os.environ["MST-STR-MON-RAWDATAPLOTS"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/rawdata'
    os.environ["MST-STR-MON-ONEFILEOMA"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/analysisresults'
    os.environ["MST-STR-MON-WEATHERRESULTS"] = os.environ["MST-STR-MON-ONEFILERESULTS"] + '/weather'
    os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] = os.environ["MST-STR-MON-RESULTS"] + '/multiplefileanalysis'
    os.environ["MST-STR-MON-MULTIPLEFILESWEATHERRESULTS"] = os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] + '/weather'
    os.environ["MST-STR-MON-REPORTS"] = os.environ["MST-STR-MON-RESULTS"] + '/reports'

# SETTING UP NAMES OF FILES
MERGING = '_files-merged'
RAW_ALL = '_raw-data-all-Ch'
RAW_CH = '_raw-data-Ch'
WIND = '_wind'
TEMPERATURE = '_temp'
PRESSURE = '_pressure'
HUMIDITY = '_humidity'
RAINRATE = '_rain-rate'
FDD = '_FDD-Chs-'
PEAKS = '_peaks'
DECIMATION = '_dec'
CHANNELS_USED = '_Ch.'
MACVALUE_ALL = '_MAC-all'
MACVALUE_SELECTED = '_MAC-selected'
SPECTOGRAM = '_spect-Ch-'
EFDD_FREQ_DAMP = '_EFDD-modal-freqs'
EFDD_MODE_SHAPE = '_EFDD-mode-shapes'
SHIFT = '_shift-Ch-'
SHIFTAVG = '_shift-avg'
LOGDEC = '_log-dec'
ZEROCROSSING = '_zero-cross'
AUTOCORREL = '_auto-correl'
BELLSHAPE = '_bell-shape'
CHISQUARE_FREQ = '_chi-square-freq'
CHISQUARE_MODE_SHAPE = '_chi-square-mode-shape'
CHISQUARE_DAMPING = '_chi-square-damp'
ALLFILES = '_all-files'
SELECTEDFILES = '_selected-files'
CHANGE_FACTOR_FREQ = '_change-factor-freq'
CHANGE_FACTOR_DAMP = '_change-factor-damp'
NUM_CORRELATION = '_num-of-correl'
TRACK_FREQ = '_track-freq'
TRACK_MODE_SHAPE = '_track-mode-shape'
TRACK_DAMP = '_track-damp'
FREQXDAMP = '_freqxdamp'
REPORTS = 'Reports-'
EMAILS_LIST = 'emails-list'
WIND_DAYS = 'wind-days'
TEMP_DAYS = 'temp-days''"""