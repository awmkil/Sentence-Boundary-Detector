# Sentence Boundary Detector

Detects sentence boundaries of HTML eBooks using a trained classifier, and put detected sentences in a ```<span>``` tag.

# Sample output

![][example]

[example]: https://github.com/awmkil/Sentence-Boundary-Detector/blob/master/.data/img.png "Test picture"

# Method

This program is based on the method described in [this article](https://www.aclweb.org/anthology/N09-2061.pdf) from ***Dan Gillick** (University of California, Berkeley)* and some chunks of code are based on his implementation.

The following features are extracted from the **Copra Brown** general text collection *(500 samples of English text)*:

- Word left of an end of sentence punctuation
- Word right of an end of sentence punctuation
- Length of the word left of an end of sentence punctuation
- Length of the word right of an end of sentence punctuation
- Capitalization of the word right of an end of sentence punctuation

I tried adding other features, like the distance between two successive end of sentence punctuations, but it did not seem to affect the accuracy of the classifier with this specific dataset.

I used a **Naive Bayes Classifier** because it gave better results than the other two recommended classifiers *(MaxEnt and SVM)*.

# Usage

## Dependencies

- nltk 3.4.5
- Numpy 1.17.4
- BeautifulSoup (```pip3 install bs4 (--user)```) 0.0.1
- tqdm 4.40.0

## Running the program
```
usage: python3 -m sentence_boundaries_detector [-h] [-l LOAD] [-t TRAIN]
                                               [-i INPUT] [-s SAVE]

Sentence boundary detector for HTML ebooks.

optional arguments:
  -h, --help            show this help message and exit
  -l LOAD, --load LOAD  Run the program using a trained model saved to your
                        computer.
  -t TRAIN, --train TRAIN
                        Train and write out serialized model.
  -i INPUT, --input INPUT
                        Segment sentences from an input HTML file.
  -s SAVE, --save SAVE  Save semented HTML file.

```

 1. First train the program using the following command:

```
python3 -m sentence_boundaries_detector -t model.pickle
```
This command will download the following resources:

- Copra Brown corpus 
- Punkt Tokenizer

And train the classifier before saving its pickled representation to ```model.pickle```.

 2. Then use this command to run the program on the provided HTML file *(or any other)*.
 
``` python3 -m sentence_boundaries_detector -l model.pickle -i sample_files/the_little_prince.html -s output_file.html ```

This will save a modified HTML file with the ```<span>``` tags to ```output_file.html```.

 3. Enjoy!

# Issues
- The segmentation is "good enough" but not great. This program is more a proof of concept than a finished product.
- The program seem to struggle in some situations (ex: '...')
- The program does not properly write the ```<span>``` tags. I think it's a text encoding issue with the ``` '<,>' ``` characters since the raw HTML shows them as ```&lt, &gt```
- Some edge cases are ignored.
