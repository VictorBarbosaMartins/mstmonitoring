import errno
import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from glob import glob

import numpy as np
import pandas as pd
from fpdf import FPDF

import definitions as names


class Reports(object):
    # This class sends reports and alerts based on the time series analysis

    # Setting directories
    def __init__(self):
        self.reportsfolder = os.environ["MST-STR-MON-REPORTS"] + '/'
        self.resultsfolder = os.environ["MST-STR-MON-MULTIPLEFILESRESULTS"] + '/'
        self.weatherresultsfolder = os.environ["MST-STR-MON-WEATHERRESULTS"] + '/'
        self.day = datetime.today().date()

        try:
            os.makedirs(self.reportsfolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def generate(self, **kwargs):

        OUTREPORT = self.reportsfolder + names.REPORTS + str(self.day) + '.pdf'
        if os.path.isfile(OUTREPORT):
            DONTRUNFLAG = 1
            print("File " + OUTREPORT + " already exists.")
            print("If you wish to generate another report, please delete the existing file")
        else:
            DONTRUNFLAG = 0
            # YELLOW LIGHT
            self.LIMITCHIFREQ1 = kwargs.get('yellowfreq', 0.001)
            self.LIMITCHIMODE1 = kwargs.get('yellowmode', 0.05)
            self.LIMITCHIDAMP1 = kwargs.get('yellowdamp', 0.002)

            # RED LIGHT
            self.LIMITCHIFREQ2 = kwargs.get('redfreq', 0.005)
            self.LIMITCHIMODE2 = kwargs.get('redmode', 0.1)
            self.LIMITCHIDAMP2 = kwargs.get('reddamp', 0.01)

            # Reading files
            dates = np.loadtxt(self.resultsfolder + "dates.txt", dtype=str, delimiter='\t')
            freqdampshift = np.loadtxt(self.resultsfolder + "freq-damping-shift.txt")
            numofcorrel = np.loadtxt(self.resultsfolder + "numofcorrelation.txt")
            chisquare = np.loadtxt(self.resultsfolder + "chisquare.txt")


            # Building data frame
            df2 = pd.DataFrame()
            df2['Date'] = dates[:-1]
            df2['Num. of correl.'] = numofcorrel
            df2['Freq-chisquare'] = np.around(chisquare[0, :], 4)
            df2['Shape-chisquare'] = np.around(chisquare[1, :], 4)
            df2['Damp-chisquare'] = np.around(chisquare[2, :], 4)

            # Flags
            Yellowflagfreq, Yellowflagshape, Yellowflagdamp = 0, 0, 0
            Yellowflag = [Yellowflagfreq, Yellowflagshape, Yellowflagdamp]
            Redflagfreq, Redflagshape, Redflagdamp = 0, 0, 0
            Redflag = [Redflagfreq, Redflagshape, Redflagdamp]

            # Writing document
            pdf = FPDF()
            pdf.add_page()
            pdf.set_xy(0, 0)
            pdf.set_font('arial', 'B', 12)
            pdf.cell(60)
            pdf.cell(75, 10, "Report at " + str(self.day) + ": MST Structure Monitoring", 0, 2, 'C')
            pdf.cell(90, 10, " ", 0, 2, 'C')
            pdf.cell(-40)
            pdf.cell(50, 10, "From " + str(dates[0]) + " to " + str(dates[-1]) + ":", 0, 2, 'C')

            pdf.cell(30, 10, 'Date', 1, 0, 'C')
            pdf.cell(50, 10, 'Freq. indicator', 1, 0, 'C')
            pdf.cell(50, 10, 'Shape indicator', 1, 0, 'C')
            pdf.cell(50, 10, 'Damp. indicator', 1, 0, 'C')
            pdf.cell(90, 10, " ", 0, 2, 'C')
            pdf.cell(-180)
            pdf.set_font('arial', '', 12)

            reportaction = [[] for i in range(2)]

            # Drawing table
            for i in range(0, len(df2)):
                pdf.cell(30, 10, '%s' % (df2['Date'].iloc[i]), 1, 0, 'C')

                if (df2['Freq-chisquare'].iloc[i] > self.LIMITCHIFREQ1) & (
                        df2['Freq-chisquare'].iloc[i] < self.LIMITCHIFREQ2):
                    pdf.set_fill_color(255, 255, 0)
                    Yellowflag[0] = 1
                elif df2['Freq-chisquare'].iloc[i] > self.LIMITCHIFREQ2:
                    pdf.set_fill_color(255, 0, 0)
                    Redflag[0] = 1
                elif str(df2['Freq-chisquare'].iloc[i]) == 'nan':
                    pdf.set_fill_color(255, 0, 0)
                    Redflag[0] = 1
                else:
                    pdf.set_fill_color(0, 255, 0)
                pdf.cell(50, 10, '%s' % (str(df2['Freq-chisquare'].iloc[i])), 1, 0, 'C', fill=True)

                if (df2['Shape-chisquare'].iloc[i] > self.LIMITCHIMODE1) & (
                        df2['Shape-chisquare'].iloc[i] < self.LIMITCHIMODE2):
                    pdf.set_fill_color(255, 255, 0)
                    Yellowflag[1] = 1
                elif df2['Shape-chisquare'].iloc[i] > self.LIMITCHIMODE2:
                    Redflag[1] = 1
                    pdf.set_fill_color(255, 0, 0)
                elif str(df2['Shape-chisquare'].iloc[i]) == 'nan':
                    pdf.set_fill_color(255, 0, 0)
                    Redflag[0] = 1
                else:
                    pdf.set_fill_color(0, 255, 0)

                pdf.cell(50, 10, '%s' % (str(df2['Shape-chisquare'].iloc[i])), 1, 0, 'C', fill=True)

                if (df2['Damp-chisquare'].iloc[i] > self.LIMITCHIDAMP1) & (
                        df2['Damp-chisquare'].iloc[i] < self.LIMITCHIDAMP2):
                    pdf.set_fill_color(255, 255, 0)
                    Yellowflag[2] = 1
                elif df2['Damp-chisquare'].iloc[i] > self.LIMITCHIDAMP2:
                    pdf.set_fill_color(255, 0, 0)
                    Redflag[2] = 1
                elif str(df2['Damp-chisquare'].iloc[i]) == 'nan':
                    pdf.set_fill_color(255, 0, 0)
                    Redflag[0] = 1
                else:
                    pdf.set_fill_color(0, 255, 0)
                pdf.cell(50, 10, '%s' % (str(df2['Damp-chisquare'].iloc[i])), 1, 2, 'C', fill=True)
                pdf.cell(-130)
                reportaction[0].append(df2['Date'].iloc[i])
                if 1 in Yellowflag:
                    reportaction[1].append(
                        'Yellow signal: There might be a moderate change in the structure. Check first if this is an outlier in the tracking curve (False alert)!')

                if 1 in Redflag:
                    reportaction[1].append(
                        'Red signal: There might be a big change in the structure. Check first if this is an outlier in the tracking curve (False alert)!')

                Yellowflag = [0, 0, 0]
                Redflag = [0, 0, 0]
            pdf.cell(90, 10, " ", 0, 2, 'C')
            pdf.cell(-20)
            pdf.set_font('arial', 'B', 12)
            pdf.cell(50, 10, "Results:", 0, 2, 'C')
            pdf.cell(90, 5, " ", 0, 2, 'C')

            # Pasting graphs
            freqgraphstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.CHISQUARE_FREQ + '*png')[0]
            pdf.image(freqgraphstring, x=15, y=None, w=180, h=120, type='', link='')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            modegraphstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.CHISQUARE_MODE_SHAPE + '*png')[
                0]
            pdf.image(modegraphstring, x=15, y=None, w=180, h=120, type='', link='')
            pdf.cell(90, 10, " ", 0, 2, 'C')
            dampgraphstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.CHISQUARE_DAMPING + '*png')[0]
            pdf.image(dampgraphstring, x=15, y=None, w=180, h=120, type='', link='')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            dampgraphstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.TRACK_FREQ + '*png')[0]
            pdf.image(dampgraphstring, x=15, y=None, w=180, h=120, type='', link='')

            #pdf.cell(90, 10, " ", 0, 2, 'C')
            #dampgraphstring = glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.EFDD_MODE_SHAPE + '*png')[0]
            #pdf.image(dampgraphstring, x=15, y=None, w=180, h=120, type='', link='')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            dampgraphstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.TRACK_DAMP + '*png')[0]
            pdf.image(dampgraphstring, x=15, y=None, w=180, h=120, type='', link='')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            numofcorrelstring = \
            glob(self.resultsfolder + '*' + dates[0] + 'until' + dates[-1] + '*' + names.NUM_CORRELATION + '*png')[0]
            pdf.image(numofcorrelstring, x=15, y=None, w=180, h=120, type='', link='')

            pdf.set_font('arial', 'B', 12)
            pdf.cell(50, 10, "Weather:", 0, 2, 'C')
            pdf.cell(90, 5, " ", 0, 2, 'C')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            winddays = \
            glob(self.weatherresultsfolder +  '*' + str(self.day)[0:4] + str(self.day)[5:7] + str(self.day)[8:10] + names.WIND_DAYS + '.png')[0]
            pdf.image(winddays, x=15, y=None, w=180, h=120, type='', link='')

            pdf.cell(90, 10, " ", 0, 2, 'C')
            tempdays = \
            glob(self.weatherresultsfolder + '*' + str(self.day)[0:4] + str(self.day)[5:7] + str(self.day)[8:10]  + names.TEMP_DAYS + '.png')[0]
            pdf.image(tempdays, x=15, y=None, w=180, h=120, type='', link='')


            """
            pdf.cell(90, 10, " ", 0, 2, 'C')
            pdf.set_font('arial', 'B', 12)
            pdf.cell(50, 10, "Message:", 0, 2, 'C')
            pdf.set_font('arial', '', 12)


            # Printing message
            print(reportaction)
            if np.size(reportaction[1]) != 0:
                for message in range(0, np.size(reportaction[1])):
                    pdf.cell(20)
                    pdf.cell(90, 10, " ", 0, 2, 'C')
                    #pdf.set_text_color(255, 0, 0)
                    pdf.cell(50, 10, 'There is a warning on ' + str(reportaction[0][message])[2:-2] + '!', 0, 2, 'C')
                    pdf.cell(90, 5, " ", 0, 2, 'C')
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(40)
                    #pdf.cell(50, 10, str(reportaction[1][message]), 0, 2, 'C')
                    #pdf.cell(23, 10, str(reportaction[1][message]), 0, 2, 'C')
                    #pdf.cell(90, 5, " ", 0, 2, 'C')

            else:
                pdf.cell(20)
                pdf.set_text_color(0, 255, 0)
                pdf.cell(50, 10, 'No big change was detected in the structure')
                # pdf.cell(90, 5, " ", 0, 2, 'C')
                pdf.set_text_color(0, 0, 0)
                pdf.cell(40)"""

            pdf.output(OUTREPORT, 'F')

        return DONTRUNFLAG

    def sendemail(self, **kwargs):
        # adapted from https://www.geeksforgeeks.org/send-mail-attachment-gmail-account-using-python/
        # Python code to illustrate Sending mail with attachments
        # from your Gmail account
        filename = kwargs.get('filename', self.reportsfolder + names.REPORTS + str(self.day) + '.pdf')
        # libraries to be imported
        fromaddr = "mst.monitoring.structure@gmail.com"
        toaddr = np.loadtxt(os.environ["MST-STR-MON-HOME"] + '/' + names.EMAILS_LIST + '.txt', dtype=str, delimiter=';')

        # instance of MIMEMultipart
        msg = MIMEMultipart()

        # storing the senders email address
        msg['From'] = fromaddr

        # storing the receivers email address

        # storing the subject
        msg['Subject'] = "Report MST STR Monitoring on " + str(self.day)

        # string to store the body of the mail
        body = "Automatically sent. Please see report attached."

        # attach the body with the msg instance
        msg.attach(MIMEText(body, 'plain'))

        # open the file to be sent
        attachment = open(filename, "rb")

        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')

        # To change the payload into encoded form
        p.set_payload((attachment).read())

        # encode into base64
        encoders.encode_base64(p)

        p.add_header('Report', "attachment; filename= %s" % filename)

        # attach the instance 'p' to instance 'msg'
        msg.attach(p)

        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()

        # Authentication
        s.login(fromaddr, "mst@adlershof")

        # Converts the Multipart msg into a string
        text = msg.as_string()

        msg['To'] = toaddr
        print("Sending email to:", toaddr)
        # sending the mail
        s.sendmail(fromaddr, toaddr, text)

        # terminating the sessions
        s.quit()


if __name__ == "__main__":
    R = Reports()
    DONTRUNFLAG = R.generate()
    if DONTRUNFLAG == True:
        print('New report not generated')
        print('Email not sended')
        print('For generating a new report, please delete the existing report')
    else:
        R.sendemail()