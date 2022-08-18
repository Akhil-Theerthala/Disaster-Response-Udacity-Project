import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merging the dataframes
    df = pd.concat([messages.set_index('id'), categories.set_index('id')], axis=1)
    
    return df


def clean_data(df):
    # Filling null values in messages in the original column
    df.original.fillna('-', inplace=True)
    
    #Seperating the categories into individual columns.
    categories = df.categories.str.split(';', expand=True)
    
    #Getting the column names
    row = categories.iloc[0,:]
    category_colnames = [x for x,y in row.str.split('-')]
    categories.columns = category_colnames
    
    #Removing the column names from the entries and changing the type to int
    for column in categories:
        categories[column] = [y for x,y in categories[column].str.split('-')]
        categories[column] = categories[column].astype('Int64')  
    
    # Check number not in (0,1) and update other value to 1
    columns=(categories.max()>1)[categories.max()>1].index
    for col in columns:
        categories.loc[categories[col]>1,col] = 1
    
    #replacing the categories column with the new columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Dropping the duplicate entries
    df.drop_duplicates(inplace=True)
    
    # remove any extra whitespace in message column
    df['message'].replace(' +', ' ', inplace=True,regex=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Figure8Messages', engine, index=False,  if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()