# Kinesis-1D-CNN
This repo contains the trained 1D CNN model, as well as the code to train your own model, for [Neurosity's kinesis EEG data set](https://github.com/neurosity/sw-kinesis-ai). The trained model within this repo has markedly better accuracy than Neurosity's random forrest model for the same data set—89.45% compared to 81.2%. Additionally, the vast majority of the missclassifications are miss of rest state labels. This more or less makes sense as people's thoughts may tend to wander a little more when "at rest" data was being gathered compared to when envisioning motor functions.

![89_45acc_500epoch_ker2_conf_mat](https://github.com/user-attachments/assets/f2b4a101-6c82-49e9-835f-4934df815e08)

**Labels:**
    0: 	Rest
    1: 	Left Arm
    4: 	Tongue
    6:	Jumping Jacks
    7:	Left Foot
    8:	Right Foot
    22:	Push
    34:	Disappear

