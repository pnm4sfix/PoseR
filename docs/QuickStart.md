# Quick Start

1) Install via pip

2) Create a decoder_config.yaml and place in folder that you want to work out of.
   See example decoder_config.yaml file here.
   https://github.com/pnm4sfix/PoseR/blob/main/src/poser/_tests/decoder_config.yml

3) From anaconda prompt, activate PoseR environment, run napari and in Plugins dropdown select PoseR

![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-LoadPlugin1.png?raw=true)


4) Load data folder containing decoder_config.yaml
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SelectProjectFolder2.png?raw=true)

5) Load DeepLabCut h5/csv file and load associated video
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SelectDLCH5File.png?raw=true)
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SelectVideoFile4.png?raw=true)

6) Select the individual you're interested in
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SelectIndividual5.png?raw=true)

7) View video and overlayed poses

8) Extract behaviour bouts
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-ExtractBouts.png?raw=true)

9) A) Cycle through and manually label behaviours

![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-LabelBehaviours7.png?raw=true)

9) B) OR use pretrained behaviour decoder to predict labels for extracted behaviours
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-AnalyseExtractedBouts.png?raw=true)

10) Save the manually or decoder labelled behaviour bouts
![alt text](https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/Workflow1-SaveLabels8.png?raw=true)