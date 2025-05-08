import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for headless environments
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

def process_participant_data(participant_number):
    print(f"\nProcessing Participant {participant_number}")
    # make a dataframe with the timestamp, the event, and the event data
    df = pd.DataFrame(columns=['timestamp', 'event', 'event_data', 'source_file', 'group', 'day'])
    
    # Get all event files from the participant's directory
    # Find the participant folder that matches the number (ignoring the name in parentheses)
    participant_dir = None
    for folder in os.listdir('No-Delete-Study'):
        if folder.startswith(f'Participant {participant_number}'):
            participant_dir = os.path.join('No-Delete-Study', folder)
            break
    
    if participant_dir is None:
        print(f"Could not find folder for Participant {participant_number}")
        return None
    
    print(f"Found directory: {participant_dir}")
    
    # Get only day folders (skip pilot folders)
    day_folders = [f for f in os.listdir(participant_dir) 
                  if os.path.isdir(os.path.join(participant_dir, f)) 
                  and f.startswith('Day')]
    day_folders.sort()  # Sort days numerically
    
    print(f"Found day folders: {day_folders}")
    
    # Keep track of cumulative time to adjust timestamps
    cumulative_time = timedelta(0)
    last_timestamp = None
    
    # Process each day folder
    for day_folder in day_folders:
        day_path = os.path.join(participant_dir, day_folder)
        # Extract day number from folder name
        day_number = int(day_folder.split()[1]) - 1  # Convert to 0-based index
        
        # Determine group from folder name (QWERTY or Circle)
        group = "Unknown"
        if "qwerty" in day_folder.lower():
            group = "Qwerty"
        elif "circle" in day_folder.lower():
            group = "Circle"
            
        print(f"Processing {day_folder} - Detected group: {group}")
            
        # Find all event files in the day folder
        event_files = [f for f in os.listdir(day_path) if f.endswith('_events.txt')]
        event_files.sort()  # Sort files by timestamp
        
        print(f"Found {len(event_files)} event files in {day_folder}")
        
        # Process each event file in the day folder
        for event_file in event_files:
            file_path = os.path.join(day_path, event_file)
            file_df = pd.DataFrame(columns=['timestamp', 'event', 'event_data', 'source_file', 'group', 'day'])
            
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        lines = line.split(',')
                        if len(lines) > 2:
                            timestamp = lines[0]
                            event = lines[1]
                            event_data = ''.join(lines[2:])
                            event_data = event_data.strip('[]').replace('\n', '').strip().upper()
                            
                            new_row = pd.DataFrame({
                                'timestamp': [timestamp], 
                                'event': [event], 
                                'event_data': [event_data],
                                'source_file': [event_file],
                                'group': [group],  # Add group to each row
                                'day': [day_number]  # Add day number to each row
                            })
                            file_df = pd.concat([file_df, new_row], ignore_index=True)
                
                if not file_df.empty:
                    # Convert timestamps to datetime
                    file_df['timestamp'] = pd.to_datetime(file_df['timestamp'], format='%H:%M:%S.%f')
                    
                    if last_timestamp is not None:
                        # Calculate the time difference needed to make this file's timestamps 
                        # continue from the last file's final timestamp
                        first_timestamp_current = file_df['timestamp'].min()
                        time_shift = (last_timestamp + timedelta(seconds=1)) - first_timestamp_current
                        
                        # Adjust timestamps by adding cumulative time
                        file_df['timestamp'] = file_df['timestamp'] + time_shift + cumulative_time
                    else:
                        # For the first file, just store its duration
                        file_df['timestamp'] = file_df['timestamp'] + cumulative_time
                    
                    # Update cumulative time and last timestamp for next file
                    file_duration = file_df['timestamp'].max() - file_df['timestamp'].min()
                    cumulative_time += file_duration
                    last_timestamp = file_df['timestamp'].max()
                    
                    # Add to main dataframe
                    df = pd.concat([df, file_df], ignore_index=True)
                    print(f"Added {len(file_df)} events from {event_file}")
            except Exception as e:
                print(f"Error processing file {event_file}: {str(e)}")

    # Sort the final dataframe by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total events processed: {len(df)}")
    if not df.empty:
        print(f"Events by group:\n{df['group'].value_counts()}")
    else:
        print("No events were processed!")

    begin_new_phrase = False
    total_phrases = 0
    phrase_data = pd.DataFrame(columns=['target_phrase', 'deletions', 'swipes', 'time_taken', 'session', 'suggestions_used', 'source_file', 'group'])
    
    # Reset variables for each new phrase
    deletions_this_phrase = 0
    swipe_count_this_phrase = 0
    suggestions_used_this_phrase = 0
    current_group = None
    time_started_phrase = None
    current_day = None
    
    for index, row in df.iterrows():
        if (row['event'].strip() == "New Phrase"):
            begin_new_phrase = True
            current_typed = ""
            target_phrase = row['event_data']
            source_file = row['source_file']
            current_group = row['group']  # Store the group for this phrase
            current_day = row['day']  # Store the day for this phrase
            
        if (begin_new_phrase and row['event'].strip() == "Swipe started"):
            # User has begun typing a new phrase
            begin_new_phrase = False
            time_started_phrase = row['timestamp']
            deletions_this_phrase = 0
            swipe_count_this_phrase = 0
            suggestions_used_this_phrase = 0
        
        if (row['event'].strip() == "Delete"):
            deletions_this_phrase += 1
        
        if (row['event'].strip() == "Swipe started"):
            swipe_count_this_phrase += 1

        if (row['event'].strip() == "Accepted Suggestion"):
            suggestions_used_this_phrase += 1

        # Check for either "Words entered correctly" or "Current Text" events
        if (row['event'].strip() in ["Words entered correctly", "Current Text"] and time_started_phrase is not None):
            # User has entered the phrase correctly or completed the phrase
            time_ended_phrase = row['timestamp']
            time_taken_phrase = time_ended_phrase - time_started_phrase
            
            new_row = pd.DataFrame({
                'target_phrase': [target_phrase], 
                'deletions': [deletions_this_phrase], 
                'swipes': [swipe_count_this_phrase], 
                'time_taken': [time_taken_phrase], 
                'session': [current_day],  # Use the day number as the session
                'suggestions_used': [suggestions_used_this_phrase],
                'source_file': [source_file],
                'group': [current_group]  # Use the stored group
            })
            phrase_data = pd.concat([phrase_data, new_row], ignore_index=True)
            total_phrases += 1
            time_started_phrase = None  # Reset for next phrase

    # add another column that is the number of characters in the target phrase
    phrase_data['target_phrase_length'] = phrase_data['target_phrase'].apply(len)

    # add another column that is the number wpm during that phrase
    phrase_data['wpm'] = (phrase_data['target_phrase_length'] / 5) / (phrase_data['time_taken'].apply(lambda x: x.total_seconds()) / 60)

    # Add participant number to the dataframe
    phrase_data['participant'] = participant_number
    
    print(f"Total phrases processed: {len(phrase_data)}")
    if not phrase_data.empty:
        print(f"Phrases by group:\n{phrase_data['group'].value_counts()}")
        print(f"Phrases by session:\n{phrase_data['session'].value_counts().sort_index()}")
    else:
        print("No phrases were processed!")

    return phrase_data

# Example usage:
if __name__ == "__main__":
    # Process multiple participants
    all_participant_data = []
    for participant_num in range(1, 8):  # Changed from range(1, 10) to range(1, 12) to include participants 10 and 11
        try:
            participant_data = process_participant_data(participant_num)
            # Convert participant number to string
            participant_data['participant'] = participant_data['participant'].astype(str)
            all_participant_data.append(participant_data)
        except FileNotFoundError:
            continue
    
    
    # Combine all participant data
    combined_data = pd.concat(all_participant_data, ignore_index=True)
    
    # Create directory for comparative plots
    os.makedirs('no_delete_study_results/comparative', exist_ok=True)
    
    # Plot average WPM comparison across participants by group
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_data, x='participant', y='wpm', hue='group')
    plt.title('WPM Distribution by Participant and Group')
    plt.xlabel('Participant Number')
    plt.ylabel('Words Per Minute')
    plt.grid(True)
    plt.savefig('no_delete_study_results/comparative/wpm_by_participant_and_group.png')
    plt.close()
    
    # Plot average deletions comparison by group
    plt.figure(figsize=(12, 6))
    avg_deletions = combined_data.groupby(['participant', 'group'])['deletions'].mean().reset_index()
    x_positions = range(len(avg_deletions))
    plt.bar(x_positions, avg_deletions['deletions'])
    plt.title('Average Deletions by Participant and Group')
    plt.xlabel('Participant Number')
    plt.ylabel('Average Deletions per Phrase')
    plt.xticks(x_positions, [f"{p} ({g})" for p, g in zip(avg_deletions['participant'], avg_deletions['group'])])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('no_delete_study_results/comparative/avg_deletions_by_participant_and_group.png')
    plt.close()
    
    # Plot average suggestions comparison by group
    plt.figure(figsize=(12, 6))
    avg_suggestions = combined_data.groupby(['participant', 'group'])['suggestions_used'].mean().reset_index()
    x_positions = range(len(avg_suggestions))
    plt.bar(x_positions, avg_suggestions['suggestions_used'])
    plt.title('Average Suggestions Used by Participant and Group')
    plt.xlabel('Participant Number')
    plt.ylabel('Average Suggestions per Phrase')
    plt.xticks(x_positions, [f"{p} ({g})" for p, g in zip(avg_suggestions['participant'], avg_suggestions['group'])])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('no_delete_study_results/comparative/avg_suggestions_by_participant_and_group.png')
    plt.close()
    
    # Plot learning curves (WPM over sessions) for all participants by group
    plt.figure(figsize=(15, 8))
    
    # Get unique participants
    participants = combined_data['participant'].unique()
    
    # Create a color map for participants
    participant_colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    
    # Store data for trendlines
    qwerty_trend_data = []
    circle_trend_data = []
    
    # Plot lines for each participant and group combination
    for i, participant in enumerate(participants):
        participant_data = combined_data[combined_data['participant'] == participant]
        
        # Plot QWERTY data
        qwerty_data = participant_data[participant_data['group'] == 'Qwerty']
        if not qwerty_data.empty:
            avg_wpm_by_session = qwerty_data.groupby('session')['wpm'].mean()
            plt.plot(avg_wpm_by_session.index, avg_wpm_by_session.values, 
                    marker='o', linestyle='-', color=participant_colors[i],
                    label=f'Participant {participant} (QWERTY)')
            # Store data for trendline
            qwerty_trend_data.append(avg_wpm_by_session)
        
        # Plot Circle data
        circle_data = participant_data[participant_data['group'] == 'Circle']
        if not circle_data.empty:
            avg_wpm_by_session = circle_data.groupby('session')['wpm'].mean()
            plt.plot(avg_wpm_by_session.index, avg_wpm_by_session.values, 
                    marker='s', linestyle='--', color=participant_colors[i],
                    label=f'Participant {participant} (Circle)')
            # Store data for trendline
            circle_trend_data.append(avg_wpm_by_session)
    
    # Calculate and plot trendlines
    if qwerty_trend_data:
        # Combine all QWERTY data
        qwerty_combined = pd.concat(qwerty_trend_data)
        qwerty_means = qwerty_combined.groupby(level=0).mean()
        # Calculate trendline
        qwerty_x = np.array(qwerty_means.index)
        qwerty_y = np.array(qwerty_means.values)
        qwerty_z = np.polyfit(qwerty_x, qwerty_y, 1)
        qwerty_p = np.poly1d(qwerty_z)
        # Plot trendline
        plt.plot(qwerty_x, qwerty_p(qwerty_x), "r-", linewidth=3, 
                label='QWERTY Trend')
    
    if circle_trend_data:
        # Combine all Circle data
        circle_combined = pd.concat(circle_trend_data)
        circle_means = circle_combined.groupby(level=0).mean()
        # Calculate trendline
        circle_x = np.array(circle_means.index)
        circle_y = np.array(circle_means.values)
        circle_z = np.polyfit(circle_x, circle_y, 1)
        circle_p = np.poly1d(circle_z)
        # Plot trendline
        plt.plot(circle_x, circle_p(circle_x), "b-", linewidth=3, 
                label='Circle Trend')
    
    plt.title('Learning Curves - WPM over Sessions', fontsize=16)
    plt.xlabel('Session Number', fontsize=14)
    plt.ylabel('Average Words Per Minute', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('no_delete_study_results/comparative/learning_curves_by_group.png')
    plt.close()

    # Plot group-level comparisons
    # Filter out expert users
    filtered_data = combined_data[combined_data['group'] != 'Expert'].copy()
    
    # Calculate averages for each participant
    participant_averages = filtered_data.groupby(['participant', 'group']).agg({
        'wpm': 'mean',
        'deletions': 'mean',
        'suggestions_used': 'mean'
    }).reset_index()
    
    # Debug print to verify participant averages
    print("\nParticipant Averages:")
    print(participant_averages)
    
    # Calculate group means for verification
    group_means = participant_averages.groupby('group').agg({
        'wpm': 'mean',
        'deletions': 'mean',
        'suggestions_used': 'mean'
    })
    print("\nGroup Means (for verification):")
    print(group_means)
    
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Define custom colors for groups
    group_colors = {'Qwerty': '#e74c3c', 'Circle': '#3498db'}
    
    # WPM comparison by group
    sns.boxplot(data=participant_averages, x='group', y='wpm', ax=ax1, linewidth=2,
                order=['Circle', 'Qwerty'],
                palette=[group_colors[g] for g in ['Circle', 'Qwerty']])
    ax1.set_title('WPM Distribution by Group', fontsize=18)
    ax1.set_ylabel('Words Per Minute', fontsize=16)
    ax1.set_xlabel('')  # Remove x-axis label
    ax1.tick_params(axis='x', labelsize=16)  # Increase x-axis tick label size
    ax1.grid(True)
    
    # Deletions comparison by group
    sns.boxplot(data=participant_averages, x='group', y='deletions', ax=ax2, linewidth=2,
                order=['Circle', 'Qwerty'],
                palette=[group_colors[g] for g in ['Circle', 'Qwerty']])
    ax2.set_title('Deletions Distribution by Group', fontsize=18)
    ax2.set_ylabel('Average Deletions per Phrase', fontsize=16)
    ax2.set_xlabel('')  # Remove x-axis label
    ax2.tick_params(axis='x', labelsize=16)  # Increase x-axis tick label size
    ax2.grid(True)
    
    # Suggestions comparison by group
    sns.boxplot(data=participant_averages, x='group', y='suggestions_used', ax=ax3, linewidth=2,
                order=['Circle', 'Qwerty'],
                palette=[group_colors[g] for g in ['Circle', 'Qwerty']])
    ax3.set_title('Suggestions Used Distribution by Group', fontsize=18)
    ax3.set_ylabel('Average Suggestions per Phrase', fontsize=16)
    ax3.set_xlabel('')  # Remove x-axis label
    ax3.tick_params(axis='x', labelsize=16)  # Increase x-axis tick label size
    ax3.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('no_delete_study_results/comparative/group_comparisons_boxplots.png')
    plt.close()

    # Create a heatmap of WPM across participants and phrases
    plt.figure(figsize=(15, 8))
    
    # Pivot the data to create a matrix of WPM values
    heatmap_data = combined_data.pivot_table(
        values='wpm',
        index='participant',
        columns=combined_data.groupby('participant').cumcount(),
        aggfunc='mean'
    )
    
    # Sort the index numerically while preserving original labels
    original_labels = heatmap_data.index
    numeric_index = pd.to_numeric(heatmap_data.index, errors='coerce')
    sorted_order = numeric_index.argsort()
    heatmap_data = heatmap_data.iloc[sorted_order]
    
    # Create the heatmap
    cbar = sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.1f',
                cbar_kws={'label': 'Words Per Minute'})
    
    # Increase colorbar label font size
    cbar.figure.axes[-1].yaxis.label.set_size(24)
    # Increase colorbar tick label size
    cbar.figure.axes[-1].tick_params(labelsize=16)
    
    plt.title('WPM Heatmap Across Participants and Phrases', fontsize=24)
    plt.xlabel('Phrase Index', fontsize=24)
    plt.ylabel('Participant', fontsize=24)
    
    # Increase tick label sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig('no_delete_study_results/comparative/wpm_heatmap.png')
    plt.close()

    # Statistical summary by group (using participant averages)
    group_stats = participant_averages.groupby('group').agg({
        'wpm': ['mean', 'std', 'count'],
        'deletions': ['mean', 'std'],
        'suggestions_used': ['mean', 'std']
    }).round(2)
    
    # Save group statistics to a CSV file
    group_stats.to_csv('no_delete_study_results/comparative/group_statistics_participant_averages.csv')
    
    # Print group statistics
    print("\nGroup Statistics (Participant Averages):")
    print(group_stats)
