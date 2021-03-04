# Spam_Email_Classification_SVM

Hi, everyone!

This project makes the email classification operation available using basic Support-Vector-Machine method.

The python .py scripts' descriptions are as followed :

1. process_email.py could process the original email dataset(SpamAssassin Public Corpus). 
There are several steps including Lower-casing, Stripping HTML, Normalizing URLs, Normalizing Email Addresses, Normalizing Numbers, Normalizing Dollars etc.

2. EmailFeatures.py : There are three functions, EmailFeatures(), readfile(), get_train_test_dataset(), all of them contribute to transfering the processed
email contents to unique array made up of number 0 and 1

3. model_train.py contains the SVM model and call the funtions to process the dataset and get a satisfying prediction results.

