'''Part 2:
The exercise of writing these functions in part 2 is helpful
practice in learning how to generate random values.
The data structure of having tuples inside of a list
is one that we'll see multiple times throughout Unit 3
So it's helpful to get familiar with it and to also be able to
simulate our own versions of lists of tuples with fake data in them.'''


import random
import pandas as pd
import numpy as np


# Part 2 Functions ==========================================================
# 1___

adjectives = ['blue', 'large', 'grainy',
              'substantial', 'potent', 'thermonuclear']
nouns = ['food', 'house', 'tree', 'bicycle', 'toupee', 'phone']


def random_phrase(list1, list2):
    '''Creates a random combination of adjectives and nouns.

    Return a single string containing a randomly selected adjective
    and noun pair.

    Parameters
    ----------
    list1 : 1-D array-like or str
            list of adjectives.
    list2 : 1-D array-like or str
            list of nouns.
    '''

    item1 = random.choice(list1)
    item2 = random.choice(list2)

    return str(item1) + ' ' + str(item2)

# print(random_phrase(adjectives, nouns))

# 2___


def random_float(min_val, max_val):
    '''Returns a random float sampled from a uniform distribution
    with a minimum value of min_val and a maximum value of max_val.

    Parameters
    ----------
    min_val : int
            lower boundary of the output interval.
    max_val : int
            upper boundary of the output interval.
    '''
    return random.uniform(min_val, max_val)

# print(random_float(2, 4))

# 3___


def random_bowling_score():
    '''Generates a random integer between low 0 (inclusive) and high 300 (exclusive).

    Parameters:
    -----------
    low : int
          lowest integer that can be drawn from the distribution.
    high : int
           largest integer that can be drawn from the distribution.
    '''
    return np.random.randint(0, 300)

# print(random_bowling_score())

# 4___


def silly_tuple():
    ''' Returns a tuple that contains three items.

    Parameters:
    -----------
    random_phrase : str
                    random adjective-noun string.
    random_float : float
                 random float representing a star-rating between 1 and 5
                 rounded to one decimal place.
    random_bowling_score : int
                           random bowling score between 0 and 300.

    '''
    my_tuple = (random_phrase(adjectives, nouns), round(
        random_float(1, 5), 1), random_bowling_score())
    return my_tuple


# print(silly_tuple())

# 5___


def silly_tuple_list(num_tuples):
    '''Returns a list filled with a designated number of silly tuples.'''

    my_list = []
    for _ in range(num_tuples):
        my_list.append(silly_tuple())
    return my_list


# print(silly_tuple_list(4))


# Part 3 Functions ==========================================================

# 1___

nan_df = pd.DataFrame({'column 0': [np.nan, 4, 3], 'column 1': [
    9, np.nan, np.nan], 'column 2': [10, 2, 2]})
test_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
null_df = pd.DataFrame(
    np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, np.nan]]))


def null_count(nan_df):
    ''' Identify missing values in the dataframe.

    Check a DataFrame for nulls and return the number of missing values.

    Parameters
    ----------
    nan_df : two-dimensional labeled data.
    '''
    return nan_df.isnull().sum().sum()


# print(null_count(nan_df))
# print(null_count(test_df))
# print(null_count(null_df))

# 2___

my_df = pd.DataFrame({'column 0': [0, 3, 6], 'column 1': [
    1, 4, 7], 'column 2': [2, 5, 8]})


def train_test_split(my_df, frac=0.8):
    '''Returns both the Training and Testing sets of a dataframe.
    Frac refers to the presence of data you would like to set aside for training.

    Parameter
    ---------

    my_df : two-dimensional labeled data
         data structure that contains labeled axis (rows and columns) just like a
         spreadsheet, a SQL table, or a dictionary of series objects.
    frac : float
           fraction of the data set aside for training
    '''
    # get random sample for training set
    train = my_df.sample(frac=frac, axis=0)
    # get everything but the random sample
    # to randomize the leftover rows
    test = my_df.drop(train.index).sample(frac=1.0)
    return train, test


# print(train_test_split(test_df))
# print(train_test_split(df, 0.2)[0])  # train set
# print(train_test_split(df, 0.2)[1]) # test set

# 3___

def randomize(my_df, seed):
    '''Randomizes all of the DataFrame cells then returns that randomized DataFrame.

    Parameter
    --------
    frac : returns a specified fraction of the dataframe in the sample.
    random_state : random seed for reproducible randomization, optional.
    '''
    random_df = my_df.sample(frac=1.0, random_state=seed)
    return random_df

# print(randomize(test_df, 10))


address_df = pd.DataFrame({'address': [
    '890 Jennifer Brooks\nNorth Janet, WY 24785',
    '8394 Kim Meadow\nDarrenville, AK 27389',
    '379 Cain Plaza\nJosephburgh, WY 06332',
    '5303 Tina Hill\nAudreychester, VA 97036'
]})


# 4___

def addy_split(addy_series):
    '''Split addresses into three columns (city, state and zip_code)

    Parameter
    ---------
    addy_series : 1-D array of string or Pandas Series.
                 Address array of strings separated by space, \n and comma
    '''

    # blank DataFrame
    new_df = pd.DataFrame()
    # lists to add info to
    city_list = []
    state_list = []
    zip_list = []

    for addy in addy_series:
        # find the values in the strings
        second_half = addy.split('\n')[1]
        city = second_half.split(',')[0]
        state = second_half.split()[-2]
        zip_code = second_half.split()[-1]
        # add the strings to lists
        city_list.append(city)
        state_list.append(state)
        zip_list.append(zip_code)

    # add the lists as new columns on the DataFrame
    new_df['city'] = city_list
    new_df['state'] = state_list
    new_df['zip'] = zip_list

    return new_df


# print(addy_split(address_df['address']))


def abbr_2_st(state_series, abbr_to_st=True):
    '''Returns a new column with the full name from a State abbreviation
    column. An input of 'FL' would return 'Florida'.

    Parameter
    ---------
    state_series : dict
                   Of the form {'state abbreviation' : 'full state name'}
    abbr_to_st = bool, default True
                When False, it takes full state names and returns state abbreviations.
                An input of 'Florida' would return 'FL'.
    '''

    state_dict = {
        'AL': 'Alabama',
        'AK': 'Alaska',
        'AZ': 'Arizona',
        'AR': 'Arkansas',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'DC': 'District Of Columbia',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'IA': 'Iowa',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'ME': 'Maine',
        'MD': 'Maryland',
        'MA': 'Massachusetts',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MS': 'Mississippi',
        'MO': 'Missouri',
        'MT': 'Montana',
        'NE': 'Nebraska',
        'NV': 'Nevada',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NY': 'New York',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VT': 'Vermont',
        'VA': 'Virginia',
        'WA': 'Washington',
        'WV': 'West Virginia',
        'WI': 'Wisconsin',
        'WY': 'Wyoming'
    }

    def abbrev_replace(abbrev):
        return state_dict[abbrev]

    def state_replace(state_name):
        reverse_state_dict = dict((v, k) for k, v in state_dict.items())
        return reverse_state_dict[state_name]

    if abbr_to_st:
        return state_series.apply(abbrev_replace)
    else:
        return state_series.apply(state_replace)


addy_states = addy_split(address_df['address'])['state']

full_state_names_column = abbr_2_st(addy_states)
# print(abbr_2_st(full_state_names_column, abbr_to_st=False))
# print(abbr_2_st(addy_states))


# 6___

def list_2_series(list_2_series, any_df):
    '''Turns a list into a series and adds it to the inputted DataFrame as a new column.

    Parameter
    ---------
    list_2_series: 1-D array
                   list of items with any data types.
    any_df : 2-D labeled data structure
             data arranged in rows and columns
    '''

    temp_df = pd.DataFrame()
    temp_df['list'] = list_2_series
    updated_df = pd.concat([any_df, temp_df], axis=1)
    return updated_df


print(list_2_series([10, 11, 12], test_df))


# 7 ___

outlier_df = pd.DataFrame(
    {'a': [1, 2, 3, 4, 5, 6],
     'b': [4, 5, 6, 7, 8, 9],
     'c': [7, 1000, 9, 10, 11, 12]})


def rm_outlier(any_df):
    '''Detects and removes outlying rows and returns and outlier cleaned DataFrame.

    Parameter
    ---------
    any_df : 2 dimensional labeled data structure
             data arranged in rows and columns.
    '''
    cleaned_df = pd.DataFrame()

    for (column_name, column_data) in any_df.iteritems():
        ''' Identifies data points outside of the interquartile
        range as outliers and gets rid of them.'''

        q_1 = column_data.quantile(0.25)
        q_3 = column_data.quantile(0.75)
        iqr = q_3 - q_1
        lower_bound = q_1 - 1.5 * iqr
        upper_bound = q_3 + 1.5 * iqr
        # print(lower_bound, upper_bound)

        mask = column_data.between(lower_bound, upper_bound, inclusive='both')
        cleaned = column_data.loc[mask]

        cleaned_df[column_name] = cleaned

    return cleaned_df
    # print(column_name, cleaned)


# print(rm_outlier(outlier_df))


# 8___

def split_dates(date_series):
    '''Splits dates of format "MM/DD/YYYY" into multiple
    columns (month, day, years) and then returns
    the same dataframe with those additional columns.

    Parameter
    ---------
    date_series : str
                  list of string dates assigned to a pandas series.
    '''

    # MM/DD/YYYY
    date_df = pd.DataFrame()

    month_list = []
    day_list = []
    year_list = []

    for date in date_series:
        month_list.append(date.split('/')[0])
        day_list.append(date.split('/')[1])
        year_list.append(date.split('/')[2])

    date_df['month'] = month_list
    date_df['day'] = day_list
    date_df['year'] = year_list

    return date_df


# print(split_dates(
#     pd.Series(
#         ['01/13/2016', '02/14/2017', '03/15/2018', '04/16/2019']
#     )))
