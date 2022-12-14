import sys
import sqlite3
import pickle
from unittest import result
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pandas as pd 
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, r2_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

sklearn.ensemble.RandomForestClassifier(verbose=1)

def load_data(database_filepath):
    '''
    load_data
    Loads data from database and returns the features, targets, columns

    input:
         database name
    outputs:
        X: messages 
        y: everything esle
        category names.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM Figure8Messages", engine)
    
    x = df.message.tolist()
    y = df.drop(columns=['message', 'original','genre'], axis=1)
    
    return x,y,y.columns


def tokenize(text):
    """
    Normalize and tokenize input text

    Inputs:
    - text: Raw text which needs to tokenized and preprocessed for the model

    Return:
    clean_tokens: tokenized text which is normalized, lemmatized, and cleaned.

    """
    text =  re.sub(r"[^a-zA-Z0-9]", " ", text.casefold())
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words('english')
    
    clean_tokens = []
    for token in tokens:
        if token not in stop:
            clean_token = lemmatizer.lemmatize(token)
            clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    pipe line construction
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    parameters = {
    'vect__max_df': [1],
    'tfidf__sublinear_tf': [True, False],
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators' : [250],
    'clf__estimator__min_samples_leaf': [4]
    }    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=True)
    
    
    return cv

def display_results(y_test, y_pred):
    '''
    Display required metrics based on the predictions and targets. 
    '''

    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()
    result=precision_recall_fscore_support(y_test, y_pred)
    f1 = 2*(result[0][0]*result[1][1])/(result[0][1]+result[1][1])

    print("Labels:", labels)
    print('------------')
    print("Accuracy:", accuracy)
    print("f1:", f1)
    print("Presicion:", result[0][0])
    print("Recall:", result[1][0])
    print('\n')

def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)
    
    # display_results(y_test, y_pred)
    for i in range(Y_test.shape[1]):
        print(Y_test.columns[i])
        print('--------------------')
        display_results(Y_test.iloc[:,i], y_pred[:,i])
        print('\n')


def save_model(model, model_filepath):
    '''
    Save the model into a pickle file.z
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()