## Testing

These are instructions for reproducing the test results in the paper.

### Step 1: Install the software

Follow installation instructions in README.md

### Step 2: Download the data

Download the data from the following link:
https://drive.google.com/drive/folders/1johMxAMzH8rN757wKdvW1nH694qHayFS?usp=sharing

### Step 3: Run the tests

1) Activate the PoseR environment:
						
	   conda activate PoseR

2) Run napari:

	   napari

3) Load the plugin:

	Select PoseR from the Plugins dropdown

![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-LoadPlugin1.png?raw=true)

4) Load the data folder containing decoder_config.yaml:

![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SelectProjectFolder2.png?raw=true)

5) Select model chkpt and click Test:

![alt text](https://github.com/pnm4sfix/PoseR/blob/main/docs/SelectChkptAndTest.png?raw=true)

6) Testing output will be in the anaconda prompt and a numpy file called predictions.npy will be saved in the chkpt folder for further evaluation.
	This prediction.npy file can then evaluated using torchmetrics.functional.classification.multiclass_accuracy
    and sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report to reproduce the accuracy, f1, recall and precision scores in the paper.

![alt text](https://github.com/pnm4sfix/PoseR/blob/main/docs/TestOutputExample.png?raw=true)