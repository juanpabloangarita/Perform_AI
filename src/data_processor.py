# Perform_AI.src.data_processor.py

from src.data_helpers import (
    clean_data_basic,
    process_date_column,
    filter_and_translate_columns,
    filter_and_translate_workouts_column,
    convert_data_types_for_activities,
    filter_workouts_df_and_remove_nans
)


class DataProcessor:
    def __init__(self, workouts=None, activities=None):
        self.workouts_df = workouts
        self.activities_df = activities
        self.workout_types_to_remove_both_dfs = ['Brick', 'Other', 'Strength', 'Day Off', 'HIIT', 'Exercice de respiration', 'Musculation']
        if workouts is not None:
            self.process_workouts()
        if activities is not None:
            self.process_activities()

    @staticmethod
    def clean_data(df):
        return clean_data_basic(df)

    @staticmethod
    def process_dates(df, date_col):
        return process_date_column(df, date_col=date_col)

    @staticmethod
    def filter_translate_cols(df, cols_map, cols_keep):
        return filter_and_translate_columns(df=df, column_mapping=cols_map, columns_to_keep=cols_keep)

    @staticmethod
    def filter_translate_workout_type_cols(df, types_to_remove, types_to_map=None):
        return filter_and_translate_workouts_column(df, workouts_to_remove=types_to_remove, sports_mapping=types_to_map)

    @staticmethod
    def convert_data_types_act(df, cols_floats):
        return convert_data_types_for_activities(df, columns_to_modify=cols_floats)

    @staticmethod
    def filter_workouts_dataframe_remove_nan(df):
        return filter_workouts_df_and_remove_nans(df)


    def process_workouts(self):
        df = self.clean_data(self.workouts_df)
        df = self.process_dates(df, 'WorkoutDay')
        columns_to_keep_workouts = ['WorkoutType', 'Title', 'WorkoutDescription', 'CoachComments',
                                'HeartRateAverage', 'TimeTotalInHours', 'DistanceInMeters', 'PlannedDuration', 'PlannedDistanceInMeters']
        df = self.filter_translate_cols(df, {}, columns_to_keep_workouts)
        df = self.filter_translate_workout_type_cols(df, self.workout_types_to_remove_both_dfs)
        df = self.filter_workouts_dataframe_remove_nan(df)
        self.workouts_df = df


    def process_activities(self):
        df = self.clean_data(self.activities_df)
        df = self.process_dates(df, 'Date')
        french_to_english = {
            'Type d\'activité': 'WorkoutType',
            'Titre': 'Title',
            'Fréquence cardiaque moyenne': 'HeartRateAverage',
            'Durée': 'TimeTotalInHours',
            'Distance': 'DistanceInMeters',
            'Calories': 'Calories'
        }
        columns_to_keep_activities = list(french_to_english.values())
        df = self.filter_translate_cols(df, french_to_english, columns_to_keep_activities)
        sports_types = {
            'Nat. piscine': 'Swim',
            'Cyclisme': 'Bike',
            'Course à pied': 'Run',
            "Vélo d'intérieur": 'Bike',
            'Cyclisme virtuel': 'Bike',
            'Course à pied sur tapis roulant': 'Run',
            'Natation': 'Swim',
        }
        df = self.filter_translate_workout_type_cols(df, self.workout_types_to_remove_both_dfs, sports_types)
        df = df.dropna()
        columns_to_float = ['HeartRateAverage', 'Calories', 'DistanceInMeters', 'TimeTotalInHours']
        df = self.convert_data_types_act(df, columns_to_float)
        self.activities_df = df
