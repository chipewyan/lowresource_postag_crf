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

<img src="https://latex.codecogs.com/gif.latex?\dfrac{\mathrm{number&tilde;of&tilde;correct&tilde;predictions}}{\mathrm{Total&tilde;number&tilde;of&tilde;predictions}}" />

The `Precision` is calculated as:
$$ \dfrac{\mathrm{number&tilde;of&tilde;correctly&tilde;predicted&tilde;X}}{\mathrm{number&tilde;of&tilde;X&tilde;predicted}}, $$
where $X$ is a label.

The `Recall` is calculated as:
$$ \dfrac{\mathrm{#~of~correctly~predicted~X}}{\mathrm{#~of~X}} $$

The `F1 score` is calculated as:
$$ \dfrac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} $$