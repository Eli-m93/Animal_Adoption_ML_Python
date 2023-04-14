import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import re
import string
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sodapy import Socrata
import xgboost as xgb
from joblib import dump, load
import datetime as date

#load new model
try:
    rfc_model = load('final_model.joblib')
except:
    rfc_model = ''

class run_model(object):

    def __init__(self, intake_df, outcome_df, model = rfc_model) -> None:
        self.intake_df = intake_df
        self.outcome_df = outcome_df
        self.model = model

    # Define way to create raw data from API
    def data_feeder(self, df):
        a = pd.DataFrame.from_records(df)
        a.columns = a.iloc[0]
        a = a[1:]
        # Annoying feature of from_records is that it replaces NaN with ''
        a = a.replace('', np.nan)
        return a

    def percent_missing(self, df):
        percent_nan = 100 * df.isnull().sum() / len(df)
        percent_nan = percent_nan[percent_nan > 0].sort_values()
        drop_list = percent_nan[(percent_nan < 2) & percent_nan != 0]
        return drop_list.index.to_list()

        # Clean Data
    def lower_case_col(self, columns):
        return columns.replace(" ", "_").lower()


    def sort_for_merge(self, var, name):
        return var.sort_values(by=[name + '_datetime']).groupby(['animal_id']).cumcount()+1


    def clean_up(self, dataframe, droplist, name):
        return (dataframe
                .rename(columns=self.lower_case_col)
                .assign(name_avail=np.where(dataframe.name.isna()  | (dataframe.name == dataframe.animal_id) , 0, 1),
                        star_in_name = np.where(dataframe.name.str.contains('[*]'), 1, 0),
                        datetime=pd.to_datetime(dataframe.datetime),
                        month=lambda x: x.datetime.dt.month
                        )
                .rename(columns={'datetime': name+'_datetime'})
                )

    def sort_for_merge(self, var):
        var['times_intaked'] = var.sort_values(
            by=([col for col in var.columns if 'datetime' in col])).groupby(['animal_id']).cumcount()+1


    def string_trim(self, df):
        # specify stop_words: words that can be ignored when simplifying our strings
        stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]

        stemmer = SnowballStemmer('english')


        def preProcessText(text):
            # lowercase and strip leading/trailing white space
            text = text.lower().strip()

            # remove HTML tags
            text = re.compile('<.*?>').sub('', text)

            # remove punctuation
            text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)

            # remove extra white space
            text = re.sub('\s+', ' ', text)

            return text


        def lexiconProcess(text, stop_words, stemmer):
            filtered_sentence = []
            words = text.split(" ")
            for w in words:
                if w not in stop_words:
                    filtered_sentence.append(stemmer.stem(w))
            text = " ".join(filtered_sentence)

            return text


        def cleanSentence(text, stop_words, stemmer):
            return lexiconProcess(preProcessText(text), stop_words, stemmer)

        # Clean the text features
        for c in ['found_location', 'breed', 'color', 'intake_condition', 'intake_type', 'sex_upon_intake']:
            df[c + '_cleaned'] = [cleanSentence(
                item, stop_words, stemmer) for item in df[c].values]
            df[c + '_cleaned'] = [cleanSentence(item, stop_words, stemmer)
                            for item in df[c].values]
        return df 

    def age_to_num(self, var):
        num, date = var.split(' ')
        num = int(num)
        if 'year' in date:
            num = num*12
        elif 'week' in date:
            num = num/4
        elif 'day' in date:
            num = num/30
        else:
            num
        if num < 0:
            num = np.nan
        return(num)


    def process_data(self):
        for i in [self.intake_df, self.outcome_df]:
            cleaned_df = self.data_feeder(i)
            drop_list = self.percent_missing(cleaned_df)
            if i == self.intake_df:
                intake_df = self.clean_up(cleaned_df, drop_list, 'intake')
                self.sort_for_merge(intake_df)
            else:
                outcome_df = self.clean_up(cleaned_df, drop_list, 'outcome')
                self.sort_for_merge(outcome_df)

        #Merge
        merged_df = intake_df.merge(outcome_df[[
        'animal_id',
        'times_intaked']],
        on=['animal_id',
            'times_intaked'],
        how='outer',
        indicator=True) 

        #Keep only what does not have an outcome
        merged_df = merged_df[merged_df._merge == 'left_only']
        
        #Drop NA
        drop_list = self.percent_missing(merged_df)
        merged_df = merged_df.dropna(subset=drop_list)

        #Make outcomes and extra vars
        merged_df = (
        merged_df
        # Few more X's
        .assign(age=merged_df.age_upon_intake.apply(self.age_to_num),
                # Let's make a "purebred" variable for dogs. This should only really matter for dogs, as for cats "shorthair mix" & "shorthair" for a cat are identical. Moreover, what is the difference between a "cow" and "cow mix?"
                purebred=np.where((merged_df.animal_type == "Dog") & ~(
                    merged_df.breed.str.contains("Mix|/")), 1, 0),
                days=lambda x:(date.datetime.today() -
                    x.intake_datetime) / np.timedelta64(1, 'D'),
                )
            )


        #Drop if age < 0 (We merged bad obs)
        merged_df = merged_df[merged_df.age > 0]



        #Clean Strings
        merged_df = self.string_trim(merged_df)

        #Specify types
        #Clean up our data
        merged_df=(merged_df
        .astype({'animal_type':'category',
        'month':'category',
        'sex_upon_intake':'category',
        'intake_condition':'category',
        'intake_type':'category',
        'found_location_cleaned':'string',
        'animal_id': 'string',
        'breed_cleaned': 'string',
        'color_cleaned': 'string',
        'purebred': 'int',
        'name_avail':'int'})
)
        #Drop any that are negitive days
        #Drop IDs that are bad
        drop_id = list(merged_df.loc[merged_df.days < 0].animal_id.unique())
        merged_df = merged_df[~merged_df['animal_id'].isin(drop_id)]

        self.merged_df = merged_df

        return merged_df


    #Import model
    def import_model(self, new_model):
        self.model = new_model
        return self.model

    def predict(self):
        self.predicted_prob = self.model.predict_proba(self.merged_df)[:,1]
        
        final_df = (pd
        .merge(self.merged_df.reset_index(), 
        pd.DataFrame(self.predicted_prob), left_index=True, right_index=True)
        .rename(columns={0:'Predicted Rehome %'})
        .drop(columns=['times_intaked', 'datetime2', "_merge"])
        .sort_values(by='Predicted Rehome %')
        )

        #drop cleaned columns
        final_df = final_df.loc[:,~final_df.columns.str.contains('_cleaned')]
        return final_df

    def run(self):
        self.process_data()
        print('Cleaning Done')
        df = self.predict()
        return df
        


        

        


