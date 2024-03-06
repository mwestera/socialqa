import pandas as pd
import json
import click
import numpy as np
import matplotlib.pyplot as plt

def create_df_from_json(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    replies_data = []
    counter=0
    # Iterate through the DataFrame to process replies
    for index, row in df.iterrows():
        row_reply_ids = []
        if (not pd.isna(row["num_comments"])):
            counter+=int(row["num_comments"])
        if 'replies' in row and row['replies'] is not None:
            for reply in row['replies']:
                replies_data.append(reply)
                row_reply_ids.append(reply['id'])
        df.at[index, 'replies'] = row_reply_ids  # Replace 'replies' with list of IDs
    print(f"number of comments {counter}")
    # Create a DataFrame from the replies data
    replies_df = pd.DataFrame(replies_data)

    # For comements-on-comments

    return df, replies_df, # Return both DataFrames for further use
def plot_questions_per_user(df):
    # Filter the DataFrame to include only rows where the 'body' contains a question mark
    question_posts = df[df['body'].str.contains('\?', na=False)]
    
    # Count the number of such posts per user
    questions_per_user = question_posts['author_id'].value_counts()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    questions_per_user.plot(kind='hist', bins=50, title='Distribution of Question Posts per User')
    plt.xlabel('Number of Question Posts')
    plt.ylabel('Number of Users')
    plt.show()
@click.command()
@click.option("--json-path", type=click.Path(dir_okay=False, exists=True), help="Path to the JSON Lines file.")
def main(json_path):
    df, replies_df= create_df_from_json(json_path)

    plot_questions_per_user(df)
# Assuming 'user_id' is the field name and 'submission' identifies the posts in 'type'
    submissions_df = df[df['type'] == 'submission']

    # Count the number of submissions per user
    submissions_per_user = submissions_df['author_id'].value_counts()

    # Calculate the average number of submissions per user
    average_submissions_per_user = submissions_per_user.mean()

    print(f"Average number of submissions per user: {average_submissions_per_user}")

    # Optionally, plot the distribution of submissions per user
    plt.figure(figsize=(10, 6))
    submissions_per_user.plot(kind='hist', bins=50, title='Distribution of Submissions per User')
    plt.xlabel('Number of Submissions')
    plt.ylabel('Number of Users')
    plt.axvline(average_submissions_per_user, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_submissions_per_user * 1.1, plt.ylim()[1] * 0.9, f'Average: {average_submissions_per_user:.2f}')
    plt.show()
    # Assuming 'user_id' is the field name and 'comment' identifies the comments in 'type'
    comments_df = df[df['type'] == 'comment']

    # Count the number of comments per user
    comments_per_user = comments_df['author_id'].value_counts()

    # Calculate the average number of comments per user
    average_comments_per_user = comments_per_user.mean()

    print(f"Average number of comments per user: {average_comments_per_user}")

    # Plot the distribution of comments per user
    plt.figure(figsize=(10, 6))
    comments_per_user.plot(kind='hist', bins=50, title='Distribution of Comments per User')
    plt.xlabel('Number of Comments')
    plt.ylabel('Number of Users')
    plt.axvline(average_comments_per_user, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_comments_per_user * 1.1, plt.ylim()[1] * 0.9, f'Average: {average_comments_per_user:.2f}')
    plt.show()

    # Calculate the length of each text in the 'body' column
    df['text_length'] = df['body'].apply(lambda x: len(x) if isinstance(x, str) else 0)


    # Plot a histogram of text lengths
    plt.figure(figsize=(10, 6))
    df['text_length'].plot(kind='hist', bins=200, title='Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()
        # Plotting the distribution of num_comments in df


    if 'score' in df.columns:
        plt.figure(figsize=(10, 6))
        df['score'].plot(kind='hist', bins=1000, title='Distribution of Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.xlim(-30, 80)  # Set the x-axis to show only scores between 0 and 100
        plt.show()

    if 'num_comments' in df.columns:
        plt.figure(figsize=(10, 6))
        df['num_comments'].plot(kind='hist', bins=30, title='Distribution of Number of Comments')
        plt.xlabel('Number of Comments')
        plt.ylabel('Frequency')
        plt.show()

    # Plotting the distribution of upvote_ratio in df
    if 'upvote_ratio' in df.columns:
        plt.figure(figsize=(10, 6))
        df['upvote_ratio'].plot(kind='hist', bins=20, title='Distribution of Upvote Ratios')
        plt.xlabel('Upvote Ratio')
        plt.ylabel('Frequency')
        plt.show()

    # Plotting the distribution of the number of replies per unique parent in replies_df
    if 'parent_id' in replies_df.columns:
        replies_per_parent = replies_df['parent_id'].value_counts()
        plt.figure(figsize=(10, 6))
        replies_per_parent.plot(kind='hist', bins=30, title='Distribution of Replies per Unique Parent')
        plt.xlabel('Number of Replies')
        plt.ylabel('Frequency')
        plt.show()
if __name__ == '__main__':
    main()
