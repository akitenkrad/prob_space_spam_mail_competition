# Spam Mail Competition from Prob Space

## Data Augmentation

1. download data into the 'data/' directory
2. clone EDA repository  
    ```bash
    $ git clone https://github.com/jasonwei20/eda_nlp.git
    $ pip install -U nltk
    $ python -c "import nltk; nltk.download('wordnet')"
    ```
3. convert the format of train_data.csv into one used in the EDA script. see README for detail.
4. run EDA  
    ```bash
    $ cd eda_nlp/
    $ python code/augment.py --input <source file name> --num_aug <number of augmented sentences per original sentence>
    ```

5. augmented data file is generated
