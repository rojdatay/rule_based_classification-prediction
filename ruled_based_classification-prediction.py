################################### RULE BASED CLASSIFICATION  ########################################

# A game company wants to create level-based new customer definitions (personas) by using some
# features ( Country, Source, Age, Sex) of its customers, and to create segments according to these new customer
# definitions and to estimate how much profit can be generated from  the new customers according to these segments.

# In this study, how to do rule-based classification, customer-based revenue calculation and prediction

#####################################################################
# Importing Libraries
#####################################################################

import pandas as pd
import matplotlib.pyplot as plt
plt.matplotlib.use('Qt5Agg')
import seaborn as sns

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', True)

# Load Dataset
df= pd.read_csv("Dataset/python_for_data_science/data_analysis_with_python/datasets/persona.csv")

#####################################################################
# Describe The Data
#####################################################################

def describe_data(df):
    print("###################### First 5 Lines ###################")
    print(df.head())
    print("###################### Last 5 Lines ###################")
    print(df.tail())
    print("###################### Types ###################")
    print(df.dtypes)
    print("######################## Shape #########################")
    print(df.shape)
    print("######################### Info #########################")
    print(df.info())
    print("######################### N/A ##########################")
    print(df.isnull().sum())
    print("######################### Quantiles  ######################")
    print(df.describe().T)

data_analysis(df)

#####################################################################
# Selection of Categorical and Numerical Variables
#####################################################################
# Define a function to perform the selection of numeric and categorical variables in the data set in a parametric way.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names
        cat_th: int, float
                class threshold for numeric but categorical variables
        car_th: int, float
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical but cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("Observations: " + str(dataframe.shape[0]))
    print("Variables: " + str(dataframe.shape[1]))
    print("cat_cols: "+ str(len(cat_cols)))
    print("num_cols: " + str(len(num_cols)))
    print("cat_but_car: " + str(len(cat_but_car)))
    print("num_but_cat: " + str(len(num_but_cat)))

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

#####################################################################
# General Exploration for Categorical Data
#####################################################################

def cat_summary(dataframe, plot=False):
    for col in cat_cols:  # cat_cols = grab_col_names(dataframe)["cat_cols"]
        print("############## Frequency of Categorical Data ########################")
        print("The unique number of " + col + ": " + str(dataframe[col].nunique()))
        print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100* dataframe[col].value_counts() / len(dataframe)}))
        if plot: # plot is True (Default)
            if dataframe[col].dtypes == "bool":  # plot function not working when data type is bool
                dataframe[col] == dataframe[col].astype(int)
                sns.countplot(x=dataframe[col], data=dataframe)
                plt.show(block=True)
            else:
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.show(block=True)


cat_summary(df, plot=True)

#####################################################################
# General Exploration for Numerical Data
#####################################################################

def num_summary(dataframe, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1]
    for col in num_cols:  # num_cols = grab_col_names(dataframe)["num_cols"]
        print("########## Summary Statistics of " + col + " ############")
        print(dataframe[num_cols].describe(quantiles).T)
        if plot:
            sns.histplot(data=dataframe, x=col)
            plt.xlabel(col)
            plt.title("The distribution of " + col)
            plt.grid(True)
            plt.show(block=True)


num_summary(df, plot=True)


#####################################################################
# Data Analysis
#####################################################################

def data_analysis(dataframe):
    # Unique Values of Source:
    print("Unique Values of Source:\n", dataframe[["SOURCE"]].nunique())

    # Frequency of Source:
    print("Frequency of Source:\n", dataframe[["SOURCE"]].value_counts())

    # Unique Values of Price:
    print("Unique Values of Price:\n", dataframe[["PRICE"]].nunique())

    #  Number of product sales by sales price:
    print("Number of product sales by sales price:\n", dataframe[["PRICE"]].value_counts())

    # Number of product sales by country:
    print("Number of product sales by country:\n", dataframe["COUNTRY"].value_counts(ascending=False, normalize=True))

    # Total & average amount of sales by country
    print("Total & average amount of sales by country:\n", dataframe.groupby("COUNTRY").agg({"PRICE": ["mean", "sum"]}))

    # Average amount of sales by source:
    print("Average amount of sales by source:\n", dataframe.groupby("SOURCE").agg({"PRICE": "mean"}))

    # Average amount of sales by source and country:
    print("Average amount of sales by source and country:\n", dataframe.pivot_table(values=['PRICE'],
                                                                                    index=['COUNTRY'],
                                                                                    columns=["SOURCE"],
                                                                                    aggfunc=["mean"]))


data_analysis(df)


#####################################################################
#  Defining Personas & Creating Segments based on Persona
#####################################################################

# Let's define new level-based customers (personas) by using Country, Source, Age and Sex.

def define_persona(dataframe):
    # rank in descending order of "PRICE".
    agg_df = dataframe.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], values="PRICE", aggfunc="mean")
    agg_df = agg_df.sort_values(by="PRICE", ascending=False)
    agg_df.reset_index(inplace=True)

    # we need to convert "AGE" variable to categorical data.
    agg_df["AGE"].astype("category")

    # use qcut function based to "AGE" variable
    age_intervals = [0, 18, 23, 30, 40, 70]
    age_labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
    agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=age_intervals, labels=age_labels,right=False)

    #  For create the character, group all the properties in the dataset :
    agg_df["customers_level_based"] = [f"{COUNTRY}_{SOURCE}_{SEX}_{AGE_CAT}" for COUNTRY, SOURCE, SEX, AGE_CAT in
                                       zip(agg_df["COUNTRY"], agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT"])]
    agg_df["customers_level_based"] = agg_df["customers_level_based"].apply(lambda x: x.upper())


    # Calculating average amount of personas:
    df_persona = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
    df_persona = df_persona.reset_index()

    return df_persona

define_persona(df)

# Let's define segment by using define_persona function
def create_segments(dataframe):
    # When we list the price in descending order, we want to express the best segment as the A segment and to define 4 segments.
    df_persona = define_persona(dataframe)
    df_persona["segment"] = pd.qcut(df_persona["PRICE"], 4, labels=["D", "C", "B", "A"])
    # df_segment = df_persona.groupby("SEGMENT").agg({"PRICE": "mean"})

    return df_persona


create_segments(df)

#####################################################################
#  Prediction
#####################################################################

def ruled_based_classification_prediction(dataframe):
    df_persona = define_persona(dataframe)
    df_segment = create_segments(dataframe)

    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANDROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE_CAT = str(input("Enter the age class (0_18/19_23/24_30/31_40/41_70):"))
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_CAT

    print(new_user)
    print("Segment:" + df_segment[df_segment["customers_level_based"] == new_user].loc[:, "segment"].values[0])
    print("Price:" + str(df_segment[df_segment["customers_level_based"] == new_user].loc[:, "PRICE"].values[0]))

    return new_user

ruled_based_classification_prediction(df)

#inputs
#Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):>? TUR
#Enter the operating system of phone (IOS/ANDROID):>? IOS
#Enter the gender (FEMALE/MALE):>? FEMALE
#Enter the age class (0_18/19_23/24_30/31_40/41_70):>? 24_30

#outputs
#TUR_IOS_FEMALE_24_30
#Segment:C
#Price:34.0
#Out[259]: 'TUR_IOS_FEMALE_24_30'



