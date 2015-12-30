## Predicting Star Ratings From Yelp Review Texts

## Synopsis

This project is part of the CS221: Artificial Intelligence class taught by Percy Liang at Stanford University.
Its aim is to predict the star rating of a Yelp business review. The data was obtained www.yelp.com/dataset_challenge.

## Notes about the code
The code is implemented using Python 2.7, and requires packages NumPy, SciPy, Matplotlib, and TensorFlow.
The NRC Lexicon file can be found at: http://www.saifmohammad.com/WebPages/lexicons.html

## Running the code

To experiment with different feature extractors, and the Linear Regression, Naive Bayes and Linear Regression models,
things should be as simple as running the file main.py from the command line:
```
python main.py data/yelp_academic_dataset_review.json
```
Similarly the Neural Network on "Review Images" can be found in Tensorflow/ReviewImageNet.py and can be run as:
```
python ReviewImageNet.py data/yelp_academic_dataset_review.json
```
