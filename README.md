# lowresource_postag_crf

This is a simple CRF-based POS tagger for low-resource languages using data from Universal Dependencies.
This repository is prepared for the presentation at the Linguistics Festival 2022.

## Requirements

Check the required packages in `requirements.txt`.
`sklearn_crfsuite` is not included in the default Python3.
An easy way to install it is just to type on the command line `pip install sklearn_crfsuite`.
The detailed documentation and friendly tutorial can be found [here](https://sklearn-crfsuite.readthedocs.io/en/latest/ "sklearn-crfsuite 0.3 documentation").

## Usage

### Clone the repository

Clone this repository to your local environment.
To do so, on your command line, move to the directory where you want to work on,
and type `git clone https://github.com/chipewyan/lowresource_postag_crf.git`.

### Prepare data

To train a model, you need to have a dataset with the CoNLL-U format.
An easy way to get data is just to go to [Universal Dependencies](https://universaldependencies.org "Universal Dependencies")
and go to a repository of a language that you want to test on.
After downloading the data, I recommend you to place the data in the same directory that you have just cloned,
because it will make it easy later for you to input file names.

### Preprocess data

The raw data do not come in the way we want,
because it may have unnecessary information or ill-formed parts.
`preprocess.py` takes the UD-format raw file that you have just downloaded and creates a training data and a test data in the `.csv` format, which is often suitable for machine learning.
Immediately after running `python preprocess.py`, you are asked to write the file name of the raw data (`.conllu` file).
Then, you are asked to write the file name that you want as the output data
(Note that the name has to end with `.csv`).
You can change the ratio of train-test splitting by changing the value of the `ratio` variable.

### Train the model

`crf.py` defines the architecture of CRF, including features, and the evaluation.
After running `python crf.py`, you are asked to write the file name of the training data.
Then, you are asked to write that of the test data, but usually you can just press enter if you created it following this documentation.

Your computer will train the model, and return the evaluation of the model with regard to the training data and the test data.
Generally, the evaluation on the training data is very good, nearly perfect;
this tells us that the model was trained well to fit the training data.
The important gist is the evaluation on the test data.
The test data is unknown at the training step, so our objective is to make the model tell the correct POS tags on the test data.
With the raw data with around 1k tokens, the F1 score would be somewhere on 70%.
With around 10k tokens, the F1 score would be somewhere up on 80%.

The `Accuracy` is calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cdfrac%7B%5Cmathrm%7BCP%7D%7D%7BN%7D" 
alt="\dfrac{\mathrm{CP}}{N}">,

where CP (Correctly Predicted) is the number of correct predictions, and 
N is the number of samples predicted.

The `Precision` is calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cdfrac%7B%5Cmathrm%7BLCP%7D%7D%7B%5Cmathrm%7BLP%7D%7D" 
alt="\dfrac{\mathrm{LCP}}{\mathrm{LP}}">

where LCP (Label Correctly Predicted) is the number of the label predicted correctly (i.e., TP: true positives),
and LP (Label predicted) is the number of predicted samples with the label (i.e., TP + FP: true positives and false positives)

The `Recall` is calculated as:
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cdfrac%7B%5Cmathrm%7BLCP%7D%7D%7B%5Cmathrm%7BL%7D%7D" 
alt="\dfrac{\mathrm{LCP}}{\mathrm{L}}">

where L is the number of samples actually with the label (i.e., TP + FN: true positives and false negatives).

The `F1 score` is calculated as:
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cdfrac%7B2+%5Ccdot+%5Cmathrm%7BPrecision%7D+%5Ccdot+%5Cmathrm%7BRecall%7D%7D%7B%5Cmathrm%7BPrecision%7D+%2B+%5Cmathrm%7BRecall%7D%7D" 
alt="\dfrac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}">