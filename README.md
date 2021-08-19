# What is this?
This branch has been forked from [release1](https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1) which was then developed into a workflow for the generation of cell infection. The workflow is described in the image below.

![Infection generation workflow](doc/flow_app_final.png "Infection generation workflow")

# Step-to-step guide of infection workflow
 This workflow is aimed for use each time a new dataset is acquired. During acquisition of the new dataset the following requirements must be met: 

* The images of the new dataset need to be in .tiff format and 

* Each image must include at least three channels: the brightfield image, the DAPI and Cy3 images. (only brightfield channel needed if we are only going to predict infection from pre-trained model *--> need to check* ).

The workflow is comprised of two main parts, testing and training:
 1. The new dataset is split into train and test sets
 2. To predict cell infection a model has been trained with 85 brightfield images and achieved a Pearson Correlation Coefficient of 0.81 on 15 test images. This pre-trained model is used to generate the infection channel for the images in the test set
 3. At this point the user's input is required to visualize results and evaluate if the pre-trained model results are satisfactory. If so, the workflow ends here (the user can then also use the model to produce infection images for the full dataset)
 4. If the results are not satisfactory and the user is not happy the second part of the workflow needs to be implemented. Here, two training steps are performed:
 
 &nbsp;&nbsp;&nbsp;&nbsp; 4.1. Fine-tune model: The training data is used to fine-tune the existing model
 
 &nbsp;&nbsp;&nbsp;&nbsp; 4.2. Train from scratch: The training data is used to train a new model from scratch
 In both of these steps the following sub-workflow is implemented:
 
 &nbsp;&nbsp;&nbsp;&nbsp; Apply k-fold cross validation on data (default: 5, can be changed in ```config```). For each fold:
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Do hyperparameter search to find the best training configurations which maximize the Pearson Correlation Coefficient (default: 100 iterations, can be changed in ```config```)
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Repeat hyperparameter search (default: 5 times, can be changed in ```config```)
 
 &nbsp;&nbsp;&nbsp;&nbsp; Compute average of best hyperparameters
 
 &nbsp;&nbsp;&nbsp;&nbsp;Train a model with average best hyperparameters
  
 5. The two models from the previous step, as well as the pre-trained model are compared with respect to the Pearson Correlation Coefficient on the test set. The outputs of the best model are stored.

# How does this work?

The user interacts with the workflow through the ```config.yaml``` file. Within are included a list of parameters necessary for training and testing on a new dataset. For example, 

For steps 1-2 of the workflow described above, the user must update the ```config``` file with her new data path (```to-do```) and any other parameter she wishes and then run the following:

```
python workflow_pt1.py
```

For steps 3-5 of the workflow described above, the user must run the following:

```
python workflow_pt2.py
```

# Label-free prediction of three-dimensional fluorescence images from transmitted light microscopy
![Combined outputs](doc/PredictingStructures-1.jpg?raw=true "Combined outputs")

## System Requirements
Installing on Linux is recommended (we have used Ubuntu 16.04).

An nVIDIA graphics card with >10GB of ram (we have used an nVIDIA Titan X Pascal) with current drivers installed (we have used nVIDIA driver version 390.48).

## Installation
### Environment setup
- Install [Miniconda](https://conda.io/miniconda.html) if necessary.
- Create a Conda environment for the platform:
```shell
conda env create -f environment.yml
```
- Activate the environment:
```shell
conda activate fnet
```
- Try executing the test script:
```shell
./scripts/test_run.sh
```
The installation was successful if the script executes without errors.

## Citation
If you find this code useful in your research, please consider citing our pre-publication manuscript:
```
@article{Ounkomol2018,
  author = {Chawin Ounkomol and Sharmishtaa Seshamani and Mary M. Maleckar and Forrest Collman and Gregory R. Johnson},
  title = {Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy},
  journal = {Nature Methods}
  doi = {10.1038/s41592-018-0111-2},
  url = {https://doi.org/10.1038/s41592-018-0111-2},
  year  = {2018},
  month = {sep},
  publisher = {Springer Nature America,  Inc},
  volume = {15},
  number = {11},
  pages = {917--920},
}
```

## Contact
Gregory Johnson  
E-mail: <gregj@alleninstitute.org>

## Allen Institute Software License
This software license is the 2-clause BSD license plus clause a third clause that prohibits redistribution and use for commercial purposes without further permission.   
Copyright © 2018. Allen Institute.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.  
3. Redistributions and use for commercial purposes are not permitted without the Allen Institute’s written permission. For purposes of this license, commercial purposes are the incorporation of the Allen Institute's software into anything for which you will charge fees or other compensation or use of the software to perform a commercial service for a third party. Contact terms@alleninstitute.org for commercial licensing opportunities.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
