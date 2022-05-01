# Cherenkov Telescope Array (CTA)
## Software development for the Health Monitoring System of the Medium-Sized Telescope (MST) Structure
This Project is for the development of a system, which is able to deliver important information about the telescope structure. The analysis and the reports are generated automatically. The system should be able to detect changes in the structure such as variations in the rope tensions and other structure damages. The local team should receive a report whenever a change is detected and should act accordingly to avoid further damages to the telescopes.

## Geeting Started
These instructions are here to guide you understanding, using and maybe
developing the system further.

### Installing
After installing the packages (Conda) get your attention to the
following steps:
#### Step 1
Open definitions.py and change the names of the environment and
global variables properly;
#### Step 2
Open emails-list.txt and register your email address and whoever
wants to receive reports from the system;
#### Step 3
Using crontab -e in the terminal, add a command to run main.py everyday
at a specific time. You must also add a command to run the data
acquisition which is by now not in this project.

## Structure
### main.py
Main file, to be ran automatically

### definitions.py
File used the environment variables and global variables are defined.

### Sample
#### convertfile.py
Used to manipulate data: convert from UDBF to ASCII and text file and
merge files.

#### weathersingle.py
It analyses one single weather data file. Produces graphs for the
weather variables, derive statistical information and apply a quality
check to the dataset.

#### weathermultiple.py
It calls weathersingle.py mutiple times and return the selected
accelerometer data to be analysed based either on the quality check or
on a period of time.

#### omasingle.py
It analyses one single accelerometer data file. Produces graphs for all
the analysis pipeline

#### omamultiple.py
It calls omasingle.py multiple times and develops a trend analyses for
the used datasets

#### reports.py
Produces a report based on the trend analyses and send an email to the
registered email addresses.

### emails-list.txt
Registered email addresses for sending the report.

### 
## What's new
### V. 0.0.3: September, 1st 2019
Victor Barbosa Martins
Finished the analysis of trends, reports and structuration of the project
To do: Documentation and raise errors in functions

### V. 0.0.2: July, 30th 2019
Victor Barbosa Martins
Finished damping branch

### V. 0.0.1: May, 22th 2019
Victor Barbosa Martins
First attempt to join pieces of codes together in a single project

## What's new
### Documentation
### Errors raising
### New features
Drift, long term trend analysis, mode shape visualization, include data
taking in the project etc.

## Authors
* **Victor Barbosa Martins** - *Initial work* 
