import pandas as pd
import json
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot_submissions_stacked_histogram_by_subreddit(df, top_n_subreddits=5, pdf=None):
    """
    Plot a stacked bar histogram showing the number of submissions over time for the top N subreddits.

    Parameters:
    - df: pandas DataFrame containing the data.
    - top_n_subreddits: Number of top subreddits to display based on total submission count.
    - pdf: PdfPages object to save the plot, if provided.
    """
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFC300', '#50C878', '#FFD700']
    
    # Ensure 'created' is a datetime column
    df['created_datetime'] = pd.to_datetime(df['created'])

    # Filter for submissions only
    submissions_df = df[df['type'] == 'submission']

    # Group by subreddit and month, then count submissions
    submissions_over_time = submissions_df.groupby(['subreddit_id', pd.Grouper(key='created_datetime', freq='M')]).size().unstack(fill_value=0)
    
    # Sum across all months to find the total submissions per subreddit, then get the top N
    top_subreddits = submissions_over_time.sum(axis=1).nlargest(top_n_subreddits).index

    # Filter the data to include only the top N subreddits
    submissions_over_time_top_n = submissions_over_time.loc[top_subreddits]

    # Plotting
    plt.figure(figsize=(12, 8))
    # Since we are plotting stacked bars, we start by plotting the bottom-most data first, then stack on top of it
    bottom = np.zeros(len(submissions_over_time_top_n.columns))
    for i, subreddit in enumerate(top_subreddits):
        # Use modulo to cycle through colors if there are more subreddits than colors
        color = colors[i % len(colors)]
        plt.bar(submissions_over_time_top_n.columns, submissions_over_time_top_n.loc[subreddit], bottom=bottom, label=subreddit, color=color,width=10)
        bottom += submissions_over_time_top_n.loc[subreddit].values

    plt.title(f'Top {top_n_subreddits} Subreddits: Submissions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Submissions')
    plt.xticks(rotation=45)
    plt.legend(title='Subreddit ID')
    plt.tight_layout()

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close()
def plot_activity_over_time_dual_axis(df, pdf=None):
    """
    Plot the count of comments and submissions over time with dual y-axes to accommodate different scales.
    """
    # Ensure 'created' is a datetime column
    df['created_datetime'] = pd.to_datetime(df['created'])
    
    # Set 'created_datetime' as the DataFrame index
    df.set_index('created_datetime', inplace=True)
    
    # Resample and count comments and submissions monthly
    comments_over_time = df[df['type'] == 'comment'].resample('M').size()
    submissions_over_time = df[df['type'] == 'submission'].resample('M').size()
    
    # Creating a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Comments', color=color)
    ax1.plot(comments_over_time.index, comments_over_time.values, label='Comments', marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Submissions', color=color)  
    ax2.plot(submissions_over_time.index, submissions_over_time.values, label='Submissions', marker='x', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # Otherwise the right y-label may be slightly clipped
    plt.title('Activity Over Time (Dual Axes)')
    if pdf:
        pdf.savefig(fig)  # Only save to PDF if a PdfPages object is provided
        plt.close(fig)
def plot_activity_over_time(df, pdf=None):
    """
    Plot the count of comments and submissions over time.
    """
    # Ensure 'created' is a datetime column
    df['created_datetime'] = pd.to_datetime(df['created'])
    
    # Set 'created_datetime' as the DataFrame index
    df.set_index('created_datetime', inplace=True)
    
    # Resample and count comments and submissions monthly
    comments_over_time = df[df['type'] == 'comment'].resample('M').size()
    submissions_over_time = df[df['type'] == 'submission'].resample('M').size()
    
    plt.figure(figsize=(12, 6))
    plt.plot(comments_over_time.index, comments_over_time.values, label='Comments', marker='o', linestyle='-', color='blue')
    plt.plot(submissions_over_time.index, submissions_over_time.values, label='Submissions', marker='x', linestyle='-', color='red')
    
    plt.title('Activity Over Time')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    
    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close()
def subm_ques_reply(df,pdf):
    df_filtered=df[df["type"]=="comment"]
    # Convert 'replies' to a boolean indicating if a comment received a reply
    df_filtered['received_reply'] = df_filtered['replies'].apply(lambda x: isinstance(x, list) and len(x) > 0)

    # Filter comments with and without question marks
    df_filtered['has_question_mark'] = df_filtered['body'].str.contains('\?')
    df_filtered['has_question_mark'] = df_filtered['has_question_mark'].fillna(False)
    # Calculate the percentages
    percentage_with_question = df_filtered[df_filtered['has_question_mark']]['received_reply'].mean() * 100
    percentage_overall = df_filtered['received_reply'].mean() * 100

    # Data for plotting
    percentages = [percentage_with_question, percentage_overall]
    labels = ['Comments with Ques. Mark', 'All Comments']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(labels, percentages, color=['skyblue', 'lightgreen'])
    plt.title('Percentage of Comments Receiving a Reply')
    plt.ylabel('Percentage')

    # Annotate bars with the percentage value
    for index, value in enumerate(percentages):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close() 
def plot_types_distr(df,pdf):
    plt.figure(figsize=(10, 6))
    # Count the number of comments and submissions per user
    comments_per_user = df[df['type'] == 'comment'].groupby('author_id').size()
    submissions_per_user = df[df['type'] == 'submission'].groupby('author_id').size()

    # Calculate the average number of comments and submissions across all users
    avg_comments_per_user = comments_per_user.mean()
    avg_submissions_per_user = submissions_per_user.mean()

    # Now to plot the distribution of these counts across users
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    n, bins, patches = ax[0].hist(comments_per_user, bins=range(int(comments_per_user.min()), int(comments_per_user.max()) + 1), alpha=0.7, color='blue')
    ax[0].set_title('Distribution of Comments per User')
    ax[0].set_xlabel('Number of Comments')
    ax[0].set_ylabel('Number of Users')
    ax[0].axvline(avg_comments_per_user, color='red', linestyle='dashed', linewidth=2)
    ax[0].text(avg_comments_per_user, max(n)*0.9, f'Avg: {avg_comments_per_user:.2f}', color='red', ha='right')

    n, bins, patches = ax[1].hist(submissions_per_user, bins=range(int(submissions_per_user.min()), int(submissions_per_user.max()) + 1), alpha=0.7, color='green')
    ax[1].set_title('Distribution of Submissions per User')
    ax[1].set_xlabel('Number of Submissions')
    ax[1].axvline(avg_submissions_per_user, color='red', linestyle='dashed', linewidth=2)
    ax[1].text(avg_submissions_per_user, max(n)*0.9, f'Avg: {avg_submissions_per_user:.2f}', color='red', ha='right')

    plt.suptitle('Distribution of User Activities')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the main title so it doesn't overlap with subplots

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close() 
def calculate_text_length(df, text_column):
    """
    Adds a new column 'sentence_count' to the DataFrame with the number of sentences
    in the specified text column. Sentences are assumed to be separated by '.', '?', or '!'.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - text_column: Name of the column containing the text whose sentence count is to be calculated.
    """
    # Define a function to count sentences
    def count_sentences(text):
        if pd.notnull(text):
            # Split the text by sentence terminators and count the chunks
            return len([sentence for sentence in text.split('.') if sentence]) + \
                   len([sentence for sentence in text.split('?') if sentence]) + \
                   len([sentence for sentence in text.split('!') if sentence])
        else:
            return 0

    df['text_length'] = df[text_column].apply(count_sentences)

def plot_text_length_distribution(df, type_value, title, pdf=None):
    """
    Plots the distribution of text lengths for a specified type of posts in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data with a 'type' column and a 'text_length' column.
    - type_value: The value in the 'type' column for which to plot the distribution (e.g., 'comment' or 'submission').
    - title: Title for the plot.
    """
    
    plt.figure(figsize=(10, 6))
    # Filter the DataFrame for the specified type
    filtered_df = df[df['type'] == type_value]
    min_text_length = filtered_df['text_length'].min()
    if min_text_length <= 0:
        min_text_length = filtered_df[filtered_df['text_length'] > 0]['text_length'].min()
        if min_text_length <= 0:
            min_text_length = 1e-1  # Arbitrary small positive number if no positive lengths

    # Ensure the maximum is greater than the minimum adjusted value
    max_text_length = max(min_text_length + 1e-1, filtered_df['text_length'].max())
    bin_edges = np.logspace(np.log10(min_text_length), np.log10(max_text_length), 50)

    plt.hist(filtered_df['text_length'], bins=bin_edges, edgecolor='k')
    # Plot the distribution
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close() 
def load_jsonl_to_dataframe(jsonl_file_path):
    """
    Load a JSON Lines file into a pandas DataFrame.
    """
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)
def plot_average_comments_per_type(df, submission_df, pdf=None):
    """
    Plot the average number of comments for a specified type from both the main and replies DataFrames.
    
    Parameters:
    - df: pandas DataFrame containing the original data with a 'type' column and a 'num_comments' column.
    - replies_df: pandas DataFrame containing the replies data with a 'num_comments' column.
    - type_value: The value in the 'type' column to filter by before calculating averages.
    - pdf: PdfPages object to save the plot, if provided.
    """
    # Filter both DataFrames for the specified type, if applicable
    # Assuming replies_df does not need filtering by 'type' or is already filtered to include relevant data

    # Ensure 'num_comments' is numeric for aggregation
    df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce')
    submission_df['num_comments'] = pd.to_numeric(submission_df['num_comments'], errors='coerce')

    # Calculate the average number of comments
    avg_comments_main = df['num_comments'].mean()
    avg_comments_subm = submission_df['num_comments'].mean()

    # Data to plot
    labels = ['Avg # Comments on User Submission', 'Avg # Comments on Parent Submission']
    averages = [avg_comments_main, avg_comments_subm]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, averages, color=['skyblue', 'lightgreen'])
    plt.title(f'Average Number of Comments')
    plt.ylabel('Average Number of Comments')
    for bar, average in zip(bars, averages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(average, 2), ha='center', va='bottom')

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close()
    else:
        plt.show()
def child_replies_df(df):
    """
    Extract replies from the DataFrame and count comments.
    """
    replies_data = []
    for index, row in df.iterrows():
        row_reply_ids = []
        if 'replies' in row and row['replies']:
            for reply in row['replies']:
                replies_data.append(reply)
                row_reply_ids.append(reply['id'])
            df.at[index, 'replies'] = row_reply_ids  # Replace 'replies' with list of IDs
    return pd.DataFrame(replies_data)
def parent_submission_df(df):
    """
    Extract replies from the DataFrame and count comments.
    """
    submission_data = []
    for index, row in df.iterrows():
        if pd.notnull(row['submission']):
            submission_data.append(row['submission'])
    return pd.DataFrame(submission_data)
def plot_percentage_of_questions(df, type_value, title, pdf=None):
    """
    Plot the percentage of posts that have a question mark in them.
    
    Parameters:
    - df: pandas DataFrame containing the data with a 'body' column.
    - title: Title for the plot.
    - pdf: PdfPages object to save the plot, if provided.
    """
    filtered_df = df[df['type'] == type_value]
    # Determine if each post contains a question mark
    if(type_value=="comment"):
        contains_question_mark = filtered_df['body'].str.contains('\?', na=False)
    elif(type_value=="submission"):
        contains_question_mark = filtered_df['selftext'].str.contains('\?', na=False)
    # Calculate the percentage of posts with and without question marks
    percentage = contains_question_mark.value_counts(normalize=True).sort_index(ascending=False) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    percentage.plot(kind='bar', color=['skyblue', 'lightgreen'])
    plt.title(title)
    plt.xlabel('Contains Question Mark')
    plt.ylabel('Percentage of Posts')
    plt.xticks(ticks=[0, 1], labels=['True', 'False'], rotation=0)  # Adjust x-ticks for clarity

    # Annotate bars with the percentage value
    for index, value in enumerate(percentage):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close()

    return

# Example usage
# Assuming 'df' is your DataFrame and it has a 'body' column
# plot_percentage_of_questions(df, 'Percentage of Posts with Question Marks')

def plot_question_sentences_per_post(df, type_value, title, pdf=None):
    """
    Plot the distribution of the number of question sentences per post, replacing sequences of 
    multiple question marks with a single question mark before counting.
    """
    plt.figure(figsize=(10, 6))
    filtered_df = df[df['type'] == type_value].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Define a function to count question sentences in a text after replacing "???" or "??" with "?"
    def count_question_sentences(text):
        if pd.notnull(text):
            # Replace sequences of multiple question marks with a single one
            text_modified = text.replace('???', '?').replace('??', '?')
            return text_modified.count('?')  # Count '?' occurrences
        return 0

    # Apply the counting function to the appropriate text column
    if type_value == "comment":
        filtered_df['question_sentences_count'] = filtered_df['body'].apply(count_question_sentences)
    elif type_value == "submission":
        filtered_df['question_sentences_count'] = filtered_df['selftext'].apply(count_question_sentences)

    # Plot the distribution of question sentence counts
    filtered_df['question_sentences_count'].plot(kind='hist', bins=50, title=title)
    plt.xlabel('Number of Question Sentences per Post')
    plt.ylabel('Frequency')

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close()



def plot_distribution(df, column_name, type_value, title, xlabel, ylabel, bins=50, xlim=None, pdf=None):
    """
    General function to plot distributions for a given DataFrame column.
    """
    filtered_df= df[df['type'] == type_value]
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    filtered_df[column_name].plot(kind='hist', bins=bins, title=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close() 
def plot_comments_per_subreddit(df, pdf):
    plt.figure(figsize=(10, 6))
    #Calculate the average number of comments per subreddit and sort
    
    avg_comments_per_subreddit = df.groupby('subreddit')['num_comments'].mean()
    sorted_avg_comments = avg_comments_per_subreddit.sort_values()


    #Generate cumulative counts for y-axis
    cumulative_counts = np.arange(1, len(sorted_avg_comments) + 1)

    # Plotting
    plt.plot(sorted_avg_comments, cumulative_counts, marker='o', linestyle='-')
    
    plt.title('Cumulative Count of Subreddits by Average Number of Comments')
    plt.xlabel('Average Number of Comments')
    plt.ylabel('Cumulative Count of Subreddits')

    # Adjust x-axis ticks based on the range of average comments
    avg_comments_min = sorted_avg_comments.min()
    avg_comments_max = sorted_avg_comments.max()
    avg_comments_range = avg_comments_max - avg_comments_min
    x_ticks_step = avg_comments_range / 10  # Adjust the denominator for more or fewer ticks
    plt.xticks(np.arange(start=avg_comments_min, stop=avg_comments_max, step=x_ticks_step))

    # Adjust y-axis ticks for cumulative count of subreddits
    y_ticks_step = max(1, len(cumulative_counts) // 10)  # Ensure at least one tick
    plt.yticks(np.arange(start=0, stop=max(cumulative_counts)+1, step=y_ticks_step))

    if pdf:
        pdf.savefig()  # Only save to PDF if a PdfPages object is provided
        plt.close() 

@click.command()
@click.option("--json-path", type=click.Path(dir_okay=False, exists=True), help="Path to the JSON Lines file.")
def main(json_path):
    print("----------------------")
    df = load_jsonl_to_dataframe(json_path)
    replies_df= child_replies_df(df) #non user replies
    submission_df = parent_submission_df(df) #non user submissions
    submission_df['body'] = submission_df['selftext']
    print(submission_df.columns)
    with PdfPages('data_plots_json.pdf') as pdf:
        # Over TIME
        plot_activity_over_time(df, pdf)
        plot_activity_over_time_dual_axis(df,pdf)
        plot_submissions_stacked_histogram_by_subreddit(df, 5, pdf)

        # Types
        plot_types_distr(df, pdf)
        subm_ques_reply(df,pdf)
        # # COMMENTS
        plot_comments_per_subreddit(submission_df,pdf)
        plot_average_comments_per_type(df, submission_df,pdf=pdf)

        #QUESTIONS/USER
        plot_question_sentences_per_post(df,"submission","Distribution of Question Posts User Submissions", pdf)
        plot_question_sentences_per_post(df,"comment" ,"Distribution of Question Posts User Comments" ,pdf)

        # QUESTION%
        plot_percentage_of_questions(df,"submission","Question% User Submissions", pdf)
        plot_percentage_of_questions(submission_df,"submission" ,"Question% Parent Submissions" ,pdf)
        plot_percentage_of_questions(df,"comment" ,"Question% Users Comments" ,pdf)
        plot_percentage_of_questions(replies_df,"comment" ,"Question% Child Comments" ,pdf)

        # TEXT LENGTHS
        calculate_text_length(df, 'body')
        calculate_text_length(submission_df, 'selftext')
        calculate_text_length(replies_df, 'body')
        plot_text_length_distribution(df, 'comment', 'Distribution of Comment Sentences User',pdf)
        calculate_text_length(df, 'selftext')
        plot_text_length_distribution(replies_df, 'comment', 'Distribution of Comment Sentences Children',pdf)
        plot_text_length_distribution(df, 'submission', 'Distribution of Submission User Sentences',pdf)
        plot_text_length_distribution(submission_df, 'submission', 'Distribution of Submission Parent Sentences',pdf)

        # SCORES
        plot_distribution(df, 'score', "submission", 'Distribution of Scores User Submission', 'Score', 'Frequency', bins=2000, xlim=(-30, 100),pdf=pdf)
        plot_distribution(submission_df, 'score', "submission", 'Distribution of Scores Parent Submission', 'Score', 'Frequency', bins=2000, xlim=(-30, 500),pdf=pdf)

        # UPVOTES RATIO
        plot_distribution(df, 'upvote_ratio', "submission", 'Distribution of Upvote Ratios User Submissions', 'Upvote Ratio', 'Frequency', bins=20,pdf=pdf)
        plot_distribution(submission_df, 'upvote_ratio', "submission", 'Distribution of Upvote Ratios Parent Submissions', 'Upvote Ratio', 'Frequency', bins=20,pdf=pdf)
        # Add more plots or analysis as required

if __name__ == '__main__':
    main()

