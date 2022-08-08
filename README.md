<h6>Please note: repository is currently Work in Progress. Code modules will be added shortly.
<h2>Welcome to ODT – a novel car-following training algorithm.<h2>
<h4>ODT (Oscillation Dynamics Training) trains an LSTM car-following model to capture platoon dynamics during oscillations.

<h3>Folder contents:

<h5>Module 1 – data pre-processing<br />
This module takes the source data and converts it to the format required by the LSTM for training.<h5>

Module 2 – pre-train model<br />
This module trains an LSTM on the data from step 1. The model state dictionary is saved for further use.

Module 3 – ODT model<br />
This module loads the pre-trained model from module 2 and further trains it, using the ODT algorithm. 

Module 4 – simulate vehicle trajectories<br />
This module simulates vehicle trajectories of a selected sub-trail using both pre-trained and ODT car-following models. Sub-trial 9 is the currently selected sub-trial.

Module 5 – results<br />
The simulated trajectories are then compared to the true data and the results reported.

<h3>Instructions:

<h6>The AstaZero test trial from the OpenACC data base was used to develop and test the ODT methodology - [link to data](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/).<br />

<h5>•	Step-1: Download AstaZero data from the link provided. 
<h6>Save this data directly in the folder whose location you want to use for the rest of the project. 
This will be your project folder. Sub-trials 1,2,5,6,7,8 & 9 are required. 
<br />Note: this project folder is where all your data and models will be saved going forward. 

<h5>•	Step-2: Download all five code modules directly into your project folder.

<h5>•	Step-3: Run ‘Module 1 – data pre-processing’ from your project folder.
<h6>Within the code, you will see a variable entitled 'working_folder'. Please change this path to reflect your designated project folder.<br />
Output: You should see a subfolder entitled: LSTM training_data. This folder should contain XXXX data.

<h5>•	Step-4: Run ‘Module 2 – pre-train model’ from your project folder. 
<h6>Within the code, you will see a variable entitled 'working_folder'. Please change this path to reflect your designated project folder.<br />
Output: You should see a subfolder entitled: Pre-trained_Model. This folder contains the pre-trained model state dictionary.

<h5>•	Step-5: Run ‘Module 3 – ODT model’ from your project folder.
<h6>Within the code, you will see a variable entitled 'working_folder'. Please change this path to reflect your designated project folder.<br />
Note: depending on the machine being used, the ODT algorithm could take several hours to complete.<br />
Output: You should see a subfolder entitled: ODT_Model. This folder contains the ODT model state dictionary.

<h5>•	Step-6: Run ‘Module 4 - simulate vehicle trajectories’ from your project folder.
<h6>Within the code, you will see a variable entitled 'working_folder'. Please change this path to reflect your designated project folder.<br />
Output: A sub-folder entitled: Simulated Trajectories should appear. This folder contains two CSV files. One for each model's simulated trajectories.

<h5>•	Step-7: Run ‘Module 5 – results’ from your project folder.
<h6>Within the code, you will see a variable entitled 'working_folder'. Please change this path to reflect your designated project folder.<br />
Output: XXX data

<h4>Results explanation:
XXX
