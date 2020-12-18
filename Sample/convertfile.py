import logging
import os
import re
import struct
from datetime import datetime, timedelta
from glob import glob
import errno
import numpy as np

import definitions as names


# AbstractSignalAnalysis
class AbstractSignalAnalysis(object):

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.n_points = None
        self.n_channels = None
        self.channel_signals = None
        self.header = None
        self.timestamps = None

    def SetUDBFData(self, timestamps, signals, header):
        assert self.n_points == None
        assert self.channel_signals == None
        assert self.header == None
        assert self.timestamps == None
        assert self.n_channels == None
        assert len(timestamps) == len(signals)
        self.n_points = len(timestamps)
        self.channel_signals = np.transpose(signals)
        assert len(self.channel_signals) == header.GetNumberOfChannels()
        self.n_channels = header.GetNumberOfChannels()
        self.timestamps = timestamps
        header.CheckHeader()
        self.header = header
        self.PrintDataSummary()

    def PrintDataSummary(self):
        self.logger.info("N points: " + str(self.n_points))
        self.logger.info("N channels: " + str((len(self.channel_signals))))
        self.logger.info("First timestamp: " + str(self.timestamps[0]))
        self.logger.info("Last timestamp: " + str(self.timestamps[-1]))
        self.header.PrintHeader()

    def __str__(self):
        string = "N points: " + str(self.n_points) + "\n"
        string += "N channels: " + str((len(self.channel_signals))) + "\n"
        string += "First timestamp: " + str(self.timestamps[0]) + "\n"
        string += "Last timestamp: " + str(self.timestamps[-1]) + "\n"
        return str


class UDBFHeader(object):
    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]  # Otherwise, no pickling possible (thread locked)
        return d

    def GetEndianPrefix(self):
        assert self.endian is not None
        if self.endian == 0:
            return "<"
        return ">"

    def GetSamplingRate(self):
        return self.sampling_rate

    def CheckHeader(self):
        if type(self.number_of_channels) is not int:
            raise TypeError
        if self.number_of_channels < 0:
            raise Exception
        assert type(self.day_factor) is float
        assert type(self.second_factor) is float
        assert self.day_factor > 0.
        assert self.second_factor > 0.
        assert type(self.start_time) is float
        assert self.start_time > 0
        if self.version != 107:
            self.logger.error(
                "UDBF version is potentially incompatible with this parser. Please check the UDBF definition, in particular the OLE_TIME_ZERO offset and the time data format.")
            assert False
        assert self.variable_names is not None
        assert self.variable_units is not None
        assert self.variable_precision is not None
        assert self.variable_types is not None
        assert self.sampling_rate is not None
        channel_arrays = [self.variable_names, self.variable_units, self.variable_precision, self.variable_types]
        for array in channel_arrays:
            assert len(array) == self.number_of_channels
        assert type(self.sampling_rate) is float
        assert self.sampling_rate > 0.

    def GetTimestamp(self, timestamp):
        day_float = float(timestamp) * self.second_factor / 86400. + self.start_time * self.day_factor
        return self.OLE_TIME_ZERO + timedelta(days=day_float)

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.OLE_TIME_ZERO = datetime(1899, 12, 30, 0, 0, 0)
        # self.OLE_TIME_ZERO = datetime(1900, 1, 1, 0, 0, 0)
        self.variable_names = None
        self.variable_units = None
        self.variable_types = None
        self.variable_precision = None
        self.sampling_rate = None
        # variable_type_conversion converts from UDBF type identifier (int), defined in UDBF sheet, to python struct data types (id, byte_length)
        self.variable_type_conversion = {1: ("?", 1), 8: ("f", 4)}

    def GetVariableType(self, channel):
        assert self.variable_types is not None
        assert type(channel) is int
        assert len(self.variable_types) > channel
        variable_type = self.variable_types[channel]
        assert variable_type in self.variable_type_conversion
        return self.variable_type_conversion[variable_type]

    def GetHeaderEndByte(self):
        assert self.header_end_byte is not None
        assert type(self.header_end_byte) is int
        assert self.header_end_byte > 0
        return self.header_end_byte

    def GetNumberOfChannels(self):
        assert self.number_of_channels is not None
        assert type(self.number_of_channels) is int
        assert self.number_of_channels > 0
        return self.number_of_channels

    def PrintHeader(self):
        self.logger.info("Endian: " + str(self.endian))
        self.logger.info("Version: " + str(self.version))
        self.logger.info("Day factor: " + str(self.day_factor))
        self.logger.info("Second factor: " + str(self.second_factor))
        self.logger.info("Start time base: " + str(self.start_time))
        self.logger.info("Sampling rate: " + str(self.sampling_rate))
        self.logger.info("Number of channels: " + str(self.number_of_channels))
        self.logger.info("Header end byte: " + str(self.header_end_byte))
        self.logger.info("Variable names: " + str(self.variable_names))
        self.logger.info("Variable units: " + str(self.variable_units))

    def GetChannelVariable(self, channel):
        assert type(channel) == int
        assert channel >= 0
        if not channel < self.number_of_channels:
            self.logger.error(
                "Requested channel " + str(channel) + ". Having only " + str(self.number_of_channels) + " channels.")
            assert False
        return self.variable_names[channel].strip()

    def GetChannelUnit(self, channel):
        assert type(channel) == int
        assert channel >= 0
        assert channel < self.number_of_channels
        return self.variable_units[channel].strip()


# UDBFParser
class UDBFParser(object):

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.raw_data = None
        self.current_pointer = 0
        self.endian_prefix = None

    def GetHeader(self):
        assert self.raw_data is not None
        self.current_pointer = 0
        header = UDBFHeader()
        header.endian = self._unpack("B", 1, False)
        self.endian_prefix = header.GetEndianPrefix()
        header.version = self._unpack("H", 2)
        vendor_length = self._unpack("H", 2)
        self.current_pointer += vendor_length
        with_checksum = self._unpack("B", 1)
        if with_checksum != 0:
            self.logger.info("This file has a checksum.")
        self.n_additional_modules = self._unpack("H", 2)
        if self.n_additional_modules != 0:
            self.logger.info(
                "This data has additional module data of length " + str(self.n_additional_modules) + " bytes.")
            self.module_type = self._unpack("H", 2)
            self.module_additional_data_struct = self._unpack("H", 2)
            add_on = []
            for c in range(self.n_additional_modules - 4):
                add_on.append(self._unpack("c", 1))
        else:
            self.module_type = -1
            self.module_additional_data_struct = -1
        header.day_factor = self._unpack("d", 8)
        time_format = self._unpack("H", 2)
        header.second_factor = self._unpack("d", 8)
        header.start_time = self._unpack("d", 8)
        header.sampling_rate = self._unpack("d", 8)
        header.number_of_channels = self._unpack("H", 2)
        self._getVariables(header)
        header.CheckHeader()
        header.PrintHeader()
        return header

    def _getVariables(self, header):
        pointer = self.current_pointer
        variable_names = []
        variable_types = []
        variable_units = []
        variable_precision = []
        for channel in range(header.number_of_channels):
            variable_name_length = self._unpack("H", 2)
            variable_name = []
            for i in range(variable_name_length):
                variable_name.append(self._unpack("c", 1).decode("UTF-8", "ignore").rstrip("\x00"))
            variable_names.append("".join(variable_name))
            data_direction = self._unpack("H", 2)
            data_type = self._unpack("H", 2)
            variable_types.append(data_type)
            field_length = self._unpack("H", 2)
            precision = self._unpack("H", 2)
            variable_precision.append(precision)
            unit_length = self._unpack("H", 2)
            unit_name = []
            for i in range(unit_length):
                unit_name.append(self._unpack("c", 1).decode("UTF-8", "ignore").rstrip("\x00"))
            variable_units.append("".join(unit_name))
            additional_data = self._unpack("H", 2)
            if additional_data != 0:
                additional_data_type = self._unpack("H", 2)
                additional_data_struct_id = self._unpack("H", 2)
                additional_data_field = []
                for i in range(additional_data - 4):
                    additional_data_field.append(self._unpack("c", 1))
        header.variable_names = variable_names
        header.variable_units = variable_units
        header.variable_types = variable_types
        header.variable_precision = variable_precision
        self.logger.debug("Found header end byte at: " + str(self.current_pointer))
        header.header_end_byte = self.current_pointer

    def _unpack(self, data_type, n_bytes, with_prefix=True):
        if with_prefix:
            assert self.endian_prefix is not None
            endian_prefix = self.endian_prefix
        else:
            endian_prefix = ""
        value = \
            struct.unpack(endian_prefix + data_type,
                          self.raw_data[self.current_pointer:self.current_pointer + n_bytes])[0]
        self.current_pointer += n_bytes
        return value

    def _checkInfile(self, infile):
        if type(infile) is not str:
            self.logger.error("Input file not provided")
            raise IOError
        if os.path.isfile(infile) is False:
            self.logger.error(str(infile) + " doesn't exist")
            raise IOError

    def SetUDBFFile(self, infile):
        # self._checkInfile(infile)
        self.logger.debug("Reading " + infile)
        try:
            data_file = open(infile, mode="rb")
            self.raw_data = data_file.read()
        finally:
            data_file.close()
        self.logger.debug("Finished reading " + infile)

    def GetSignal(self):
        header = self.GetHeader()
        header_end_byte = header.GetHeaderEndByte()
        number_of_channels = header.GetNumberOfChannels()
        event_pointer = self._getSignalStartByte(self.raw_data, header_end_byte)
        self.current_pointer = event_pointer
        event_length = 8
        for channel in range(header.GetNumberOfChannels()):
            event_length += header.GetVariableType(channel)[1]
        timestamps = []
        signals = []

        self.logger.debug("Reading signal data ...")
        while self.current_pointer + event_length < len(self.raw_data):
            data_pointer = 0
            timestamp = self._unpack("Q", 8)
            timestamps.append(header.GetTimestamp(timestamp))
            event_signal_data = []
            for channel in range(header.GetNumberOfChannels()):
                variable_type = header.GetVariableType(channel)
                assert len(variable_type) == 2
                channel_data = self._unpack(variable_type[0], variable_type[1])
                event_signal_data.append(channel_data)
            signals.append(event_signal_data)

        self.logger.debug("Finished with reading signal data.")

        return (timestamps, signals)

    def _getSignalStartByte(self, data, header_end_byte):
        """
        From UDBF data sheet:
        8.6.2.2
        Separation Chars:
        There are separation characters inserted. At least 8 pieces and maximal as many as needed so that the
        next valid data byte is written to a 16 bytes aligned address.
        """
        for i in range(header_end_byte + 8, len(data)):
            if i % 16 == 0:
                self.logger.debug("Found signal start byte at: " + str(i) + " from " + str(len(data)) + " bytes")
                return i
        self.logger.error("Couldn't find signal start byte.")
        assert False


class UDBFConverter(AbstractSignalAnalysis):

    def __init__(self, ignore_channels=[0]):
        AbstractSignalAnalysis.__init__(self)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.ignore_channels = ignore_channels
        self.logger.debug("Ignoring channels " + str(self.ignore_channels))
        self.ignore_channels = ignore_channels

    def _getChannelSignal(self, channel):
        assert type(channel) == int
        assert type(self.n_channels) == int
        assert self.n_channels >= 0
        if self.n_channels == 0:
            self.logger.warn("No channels found")
        return self.channel_signals[channel]

    def Convert(self, outfile, channels, header):
        # if os.path.isfile(outfile):
        # raise IOError("File " + outfile + " already exists.")
        amplitude_array = []
        for channel in channels:
            signal = self._getChannelSignal(channel)
            amplitude_array.append(signal)
        fout = open(outfile, "a+")
        fout.write("Sampling frequency: " + str(header.GetSamplingRate()) + "\n")
        line = ""
        for i in range(len(amplitude_array)):
            line += str(header.GetChannelVariable(i).replace(" ", "_"))
            line += " "
        line += "\n"
        fout.write(line)
        for i in range(len(amplitude_array[0])):
            line = ""
            for k in range(len(amplitude_array)):
                line += str(amplitude_array[k][i])
                line += " "
            line += "\n"
            fout.write(line)
        fout.close()


def _prepare(IN, OUT):
    udbf_parser = UDBFParser()
    udbf_parser.SetUDBFFile(IN)
    header = udbf_parser.GetHeader()
    (timestamps, signals) = udbf_parser.GetSignal()
    CHANNELS = [int(channel) for channel in range(header.GetNumberOfChannels())]
    return (header, timestamps, signals, CHANNELS, OUT)


# VICTOR's code starts here (before -> Gerrit)

class Manipulatedata(object):
    # This class is meant to convert data files, manipulate and merge them
    def __init__(self, **kwargs):
        self.path = os.environ["MST-STR-MON-HOME"] + '/'
        self.datafolder = os.environ["MST-STR-MON-DATA"] + '/'
        self.datafolderresults = os.environ["MST-STR-MON-DATA-CONVERTED"] + '/'
        self.numofchannels = kwargs.get('numofchindata', 24)

        try:
            os.makedirs(self.datafolderresults)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def convertascii(self, IN, OUT):
        if os.path.isfile(OUT):
            print("File " + OUT + " already exists.")
        else:
            (header, timestamps, signals, CHANNELS, OUT) = _prepare(IN, OUT)
            converter = UDBFConverter()
            converter.SetUDBFData(timestamps, signals, header)
            converter.Convert(OUT, CHANNELS, header)
            print("Finished conversion to ASCII.")

    def converttxt(self, IN, OUT):

        if os.path.isfile(OUT):
            print("File " + OUT + " already exists.")
        else:
            rawdata = np.loadtxt(IN, delimiter='\t', dtype=str)
            rawdata = rawdata[2:]
            size = np.size(rawdata, axis=0)
            array = np.zeros((size, self.numofchannels))
            for i in range(size):
                text = [x.strip() for x in rawdata[i].split(' ')]
                text = text[2:]
                if text[-1] == '':
                    text2 = text[:-1]
                else:
                    text2 = text
                array[i] = text2
            outarray = array
            np.savetxt(OUT, outarray)
            print("Finished conversion to TXT.")

    def convertallascii(self, **kwargs):
        self.searchstring = kwargs.get('searchstring', "*")

        self.datfilenames = np.sort(glob(self.datafolder + self.searchstring + ".dat"))
        for filename in self.datfilenames:
            print(filename)
            outfileascii = self.datafolderresults + os.path.basename(filename)[:-4] + '.ascii'
            Manipulatedata.convertascii(self, IN=filename, OUT=outfileascii)
        print("Finished conversion to ASCII of all files!")

    def convertalltxt(self, **kwargs):
        self.searchstring = kwargs.get('searchstring', "*")
        self.asciifilenames = np.sort(glob(self.datafolder + self.searchstring + "*.ascii"))
        for filename in self.asciifilenames:
            outfiletxt = self.datafolder + os.path.basename(filename)[:-4] + '.txt'
            Manipulatedata.converttxt(self, IN=filename, OUT=outfiletxt)
        print("Finished conversion to TXT of all files!")

    def mergefiles(self, date, OUT, **kwargs):
        rangeoffiles = kwargs.get('rangeoffiles', [0, -1])
        self.searchstring = kwargs.get('searchstring', "structure*6.100*")
        self.fnames = np.sort(glob(self.datafolderresults + self.searchstring + date + "*.ascii"))

        if os.path.isfile(OUT):
            print("File " + OUT + " already exists.")
        else:

            # if rangeoffiles[1] == -1:
            # self.fnames = self.fnames[rangeoffiles[0]:]
            # else:
            # self.fnames = self.fnames[rangeoffiles[0]:rangeoffiles[1]]
            outarray = 0
            counter = 0
            for file in self.fnames[rangeoffiles[0]:rangeoffiles[1]]:
                rawdata = np.loadtxt(file, delimiter='\t', dtype=str)
                rawdata = rawdata[2:]
                size = np.size(rawdata, axis=0)
                array = np.zeros((size, self.numofchannels))

                for line in range(size):
                    text = [x.strip() for x in rawdata[line].split(' ')]
                    text = text[2:]
                    if text[-1] == '':
                        text2 = text[:-1]
                    else:
                        text2 = text
                    array[line] = text2

                if (counter == 0):
                    np.resize(outarray, np.shape(array))
                    outarray = array
                else:
                    outarray = np.append(outarray, array, axis=0)
                counter = counter + 1
                print(np.shape(outarray))
            self.redate = (re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(self.fnames[0])))
            initialdate = str(self.redate.group())
            # np.savetxt(self.datafolderresults + os.path.basename(self.fnames[0])[:85] + '_' + initialdate + '_' + str(np.size(self.fnames[rangeoffiles[0]:rangeoffiles[1]])) + names.MERGING + '.txt', outarray)
            np.savetxt(OUT, outarray)
            print("Merging files for one day finished!")

    def mergefilesall(self, dates, **kwargs):
        rangeoffiles = kwargs.get('rangeoffiles', [0, -1])
        searchstring = kwargs.get('searchstring', "structure*6.100*")

        self.initialdate = datetime(int(dates[0][:4]), int(dates[0][5:7]), int(dates[0][8:10]))
        self.finaldate = datetime(int(dates[1][:4]), int(dates[1][5:7]), int(dates[1][8:10]))
        self.asciifilenames = np.sort(glob(self.datafolderresults + searchstring + "*.ascii"))
        print(self.asciifilenames)
        self.sizeoffiles = np.size(self.asciifilenames)
        self.datestring = np.zeros((self.sizeoffiles)).astype(str)
        self.localdates = np.zeros((self.sizeoffiles)).astype(datetime)

        for eachdate in range(self.sizeoffiles):
            self.redate = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(self.asciifilenames[eachdate]))
            self.datestring[eachdate] = str(self.redate.group())
            self.localdates[eachdate] = datetime(int(self.datestring[eachdate][0:4]),
                                                 int(self.datestring[eachdate][5:7]),
                                                 int(self.datestring[eachdate][8:10]))

        self.localdates = np.unique(np.sort(self.localdates))
        self.datestring, countfilesperday = np.unique(np.sort(self.datestring), return_counts=True)
        print(countfilesperday)

        counter = 0
        for date in self.localdates:
            print(date)
            if date >= self.initialdate and date <= self.finaldate:
                OUT = self.datafolderresults + self.datestring[counter] + '_' + str(
                    np.size(range(rangeoffiles[0], rangeoffiles[1]))) + names.MERGING + '.txt'
                Manipulatedata.mergefiles(self, date=self.datestring[counter], OUT=OUT, rangeoffiles=rangeoffiles,
                                          searchstring=searchstring)
            counter = counter + 1
        print("Merging files for all days finished!")


if __name__ == "__main__":
    A = Manipulatedata()
    A.convertallascii(searchstring='*1.6.100*')
    today = str(datetime.today())[:10]
    A.mergefilesall(rangeoffiles=[0, 15], dates=['2019-08-16', '2020-02-01'])