import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report

folder_path = 'DATA'

datasets = ['RB_DATA_1993.csv', 'RB_DATA_1994.csv', 'RB_DATA_1995.csv', 'RB_DATA_1996.csv', 'RB_DATA_1997.csv', 'RB_DATA_1998.csv', 'RB_DATA_1999.csv', 'RB_DATA_2000.csv', 'RB_DATA_2002.csv', 'RB_DATA_2003.csv', 'RB_DATA_2004.csv', 'RB_DATA_2005.csv', 'RB_DATA_2006.csv', 'RB_DATA_2007.csv', 'RB_DATA_2008.csv', 'RB_DATA_2009.csv','RB_DATA_2010.csv', 'RB_DATA_2011.csv', 'RB_DATA_2012.csv', 'RB_DATA_2013.csv', 'RB_DATA_2014.csv', 'RB_DATA_2015.csv','RB_DATA_2016.csv', 'RB_DATA_2017.csv', 'RB_DATA_2018.csv', 'RB_DATA_2019.csv', 'RB_DATA_2020.csv', 'RB_DATA_2021.csv', 'RB_DATA_2022.csv', 'RB_DATA_2023.csv' ]  # List of stored datasets

all_data = []

player_names_all = []

results = []

# Lists to store training and testing results
train_results = []
test_results = []

# Loop through datasets
for dataset in datasets:
    # Extract the year from the dataset name
    year = dataset.split('_')[-1].split('.')[0]
    file_path = os.path.join(folder_path, dataset)
    df = pd.read_csv(file_path)
    
    df['Awards'] = df['Awards'].fillna(0).astype(int)  # Clean 'Awards' column

    # Check for Heisman winner (only one should be present)
    heisman_winners = df[df['Awards'] == 1]
    if len(heisman_winners) != 1:
        print(f"Warning: More than one or no Heisman winner found in {year} dataset!")
        continue  # Skip this dataset if there's no valid Heisman winner
    
    # Track player names
    player_names_all.append(df['Player'].values)
    
    # Prepare features and target
    X = df.drop(['Awards', 'Player'], axis=1).select_dtypes(include=['number'])
    y = df['Awards']

    df = df.reset_index(drop=True)
    
    # Check if this year should be used for testing or training
    if year[-1] in ['0', '5']:  # Testing set for years ending in '0' or '5'
        # Split data into test set
        X_test = X
        y_test = y

        # Scale the data for testing
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Build the model
        model = tf.keras.Sequential([ 
            tf.keras.layers.Dense(16, activation='relu', input_shape=(X_test_scaled.shape[1],)), 
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid') 
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model on the training set from previous years
        if len(train_results) > 0:
            X_train_scaled = np.concatenate([result['X_train_scaled'] for result in train_results], axis=0)
            y_train = np.concatenate([result['y_train'] for result in train_results], axis=0)
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Predict on the test set and get probabilities
        y_pred_prob = model.predict(X_test_scaled)
        
        # Retrieve player names for the test set
        test_player_names = df.loc[X_test.index, 'Player'].values

        # Prepare the test data for comparison
        test_data = X_test.copy()
        test_data['Actual Heisman'] = y_test.values
        test_data['Player'] = test_player_names
        test_data['Predicted Probability'] = y_pred_prob 
        
        # Sort by predicted probability to get the top 3 predicted winners
        top_3_predicted = test_data.nlargest(3, 'Predicted Probability')[['Player', 'Predicted Probability']]

        # Get player statistics for the top 3 by looking them up in the original dataset
        top_3_stats = df[df['Player'].isin(top_3_predicted['Player'].values)][[
            'Player', 'Team', 'Conf', 'G', 'Att', 'Yds', 'Y/A', 'TD', 'Y/G', 'Rec', 'Yds.1', 
            'Y/R', 'TD.1', 'Y/G.1', 'Plays', 'Yds.2', 'Avg', 'TD.2'
        ]]

        top_3_stats['-9999'] = top_3_stats['Player'].str.lower().str.replace(' ', '-')
        
        # Save the top 3 players and their stats
        output_file_path = os.path.join('DATA', f'top_3_players_{year}.csv')
        top_3_stats.to_csv(output_file_path, index=False)

        # Check if there is at least one actual Heisman winner
        actual_winner = test_data[test_data['Actual Heisman'] == 1]
        
        if actual_winner.empty:
            print(f"Warning: No actual Heisman winner found for {year}. Skipping this dataset.")
            continue  # Skip this year if there's no actual winner

        # If there's an actual winner, get the name
        actual_winner_name = actual_winner['Player'].values[0]
        print(f"Actual Heisman Winner for {year}: {actual_winner_name}") 
        
        # Check if one of the predicted winners matches the actual winner
        predicted_winners = top_3_predicted['Player'].values
        match = 'Yes' if actual_winner_name in predicted_winners else 'No'

        # Store the test results for this year
        test_results.append({
            'Year': year,
            'Predicted Winners': ', '.join(predicted_winners),
            'Actual Winner': actual_winner_name,
            'Match': match
        })

    else:
        # Split data into training set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data for training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Store training results for later use
        train_results.append({
            'X_train_scaled': X_train_scaled,
            'y_train': y_train
        })
        
test_results_df = pd.DataFrame(test_results)

# Output the results for test set
print(test_results_df)
