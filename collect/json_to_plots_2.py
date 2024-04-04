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
from scipy.stats import gaussian_kde
import seaborn as sns
import re
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
    
def plot_posts_overview(df, pdf):
    """
    Creates a visual representation of total posts (comments + submissions) per category,
    saves the plot into a PDF.
    """
    # Ensure 'created' is a datetime column
    df['created_datetime'] = pd.to_datetime(df['created'])

    # Identify the top 3 subreddits
    top_subreddits = df['subreddit'].value_counts().nlargest(3).index.tolist()
    categories = top_subreddits + ['Others']

    # Initialize a dictionary to hold total posts counts
    total_posts_per_category = {}

    for subreddit in categories:
        if subreddit != 'Others':
            total_posts = len(df[df['subreddit'] == subreddit])
        else:
            total_posts = len(df[~df['subreddit'].isin(top_subreddits)])
        total_posts_per_category[subreddit] = total_posts

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(total_posts_per_category.keys())
    posts_counts = list(total_posts_per_category.values())
    bars = ax.bar(categories, posts_counts, color=['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'])

    # Annotate bars with the exact counts
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel('Category')
    ax.set_ylabel('Total Posts')
    ax.set_title('Total Posts per Category (Top 3 and Others)')
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a PDF
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)


def plot_activity_over_time(df, pdf=None):
    """
    Plots either comments or submissions over time for the top 3 subreddits and 'Others'.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - activity_type (str): 'comment' or 'submission' to specify the type of activity to plot.
    - pdf (PdfPages, optional): A PdfPages object to save the plot into a PDF. 
    """

    # Convert 'created' to datetime and sort
    df['created_datetime'] = pd.to_datetime(df['created'])
    df.sort_values('created_datetime', inplace=True)

    # Find the top 3 subreddits
    top_subreddits = df['subreddit'].value_counts().nlargest(3).index

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for subreddit in list(top_subreddits) + ['Others']:
        if subreddit == 'Others':
            subset = df[(~df['subreddit'].isin(top_subreddits))]
        else:
            subset = df[(df['subreddit'] == subreddit)]
        
        # Resample and count over time
        activity_over_time = subset.resample('ME', on='created_datetime').size()
        # Convert the index to a numpy array and reshape
        index_array = activity_over_time.index.to_numpy()[:, None]  # This is just an example; actual reshaping depends on your needs

        # Now plot using the modified index
        ax.plot(index_array, activity_over_time.values, label=subreddit)  # Note: .values to ensure compatibility


    # Customize the plot
    ax.set_title(f'Activity over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()

    # Save or show the plot
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

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
def plot_types_distr(df, pdf=None):
    plt.figure(figsize=(10, 6))
    # Count the number of comments and submissions per user
    comments_per_user = df[df['type'] == 'comment'].groupby('author_id').size()
    submissions_per_user = df[df['type'] == 'submission'].groupby('author_id').size()

    # Calculate the average number of comments and submissions across all users
    avg_comments_per_user = comments_per_user.mean()
    avg_submissions_per_user = submissions_per_user.mean()

    # Now to plot the distribution of these counts across users, but horizontally
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Prepare bins for comments
    comment_bins = np.arange(comments_per_user.min(), comments_per_user.max() + 2) - 0.5
    ax[0].hist(comments_per_user, bins=comment_bins, orientation='horizontal', alpha=0.7, color='blue')
    ax[0].axhline(avg_comments_per_user, color='red', linestyle='dashed', linewidth=2)
    ax[0].text(max(ax[0].get_xlim()) * 0.9, avg_comments_per_user, f'Avg: {avg_comments_per_user:.2f}', color='red', va='center')
    ax[0].set_title('Distribution of Comments per User')
    ax[0].set_ylabel('Number of Comments')
    ax[0].set_xlabel('Number of Users')

    # Prepare bins for submissions
    submission_bins = np.arange(submissions_per_user.min(), submissions_per_user.max() + 2) - 0.5
    ax[1].hist(submissions_per_user, bins=submission_bins, orientation='horizontal', alpha=0.7, color='green')
    ax[1].axhline(avg_submissions_per_user, color='red', linestyle='dashed', linewidth=2)
    ax[1].text(max(ax[1].get_xlim()) * 0.9, avg_submissions_per_user, f'Avg: {avg_submissions_per_user:.2f}', color='red', va='center')
    ax[1].set_title('Distribution of Submissions per User')
    ax[1].set_xlabel('Number of Users')

    plt.suptitle('Distribution of User Activities')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the main title so it doesn't overlap with subplots

    if pdf:
        pdf.savefig(fig)  # Save the current figure into the PDF if a PdfPages object is provided
        plt.close(fig)

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

def plot_text_length_distribution(df_list, title, pdf=None):
    """
    Plots the distribution of text lengths across multiple DataFrames.
    
    Parameters:
    - df_list: List of pandas DataFrame containing the data with a 'text_length' column.
    - title: Title for the plot.
    - pdf: PdfPages object to save the plot, if provided.
    """
    plt.figure(figsize=(10, 6))
    
    # Aggregate DataFrames if there are multiple
    if len(df_list) > 1:
        aggregated_df = pd.concat(df_list, ignore_index=True)
    else:
        aggregated_df = df_list[0]
    
    # Ensure there are valid text lengths to plot
    if 'text_length' in aggregated_df.columns and not aggregated_df['text_length'].empty:
        min_text_length = aggregated_df['text_length'].min()
        if min_text_length <= 0:
            min_text_length = aggregated_df[aggregated_df['text_length'] > 0]['text_length'].min()
            if min_text_length <= 0:
                min_text_length = 1e-1  # Arbitrary small positive number if no positive lengths

        # Ensure the maximum is greater than the minimum adjusted value
        max_text_length = max(min_text_length + 1e-1, aggregated_df['text_length'].max())
        bin_edges = np.logspace(np.log10(min_text_length), np.log10(max_text_length), 50)

        # Plot the distribution
        plt.hist(aggregated_df['text_length'], bins=bin_edges, edgecolor='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(title)
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        
        if pdf:
            pdf.savefig()  # Only save to PDF if a PdfPages object is provided
            plt.close() 
    else:
        print("No 'text_length' column found or the column is empty.")
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
def plot_percentage_of_questions(df_list, title, pdf=None):
    """
    Plot the percentage of posts that have a question mark in them, aggregating data
    from multiple DataFrames if provided.
    
    Parameters:
    - df_list: List of pandas DataFrame containing the data with a 'body' column.
    - title: Title for the plot.
    - pdf: PdfPages object to save the plot, if provided.
    """
    # Aggregate DataFrames if there are multiple
    if len(df_list) > 1:
        aggregated_df = pd.concat(df_list, ignore_index=True)
    else:
        aggregated_df = df_list[0]

    # Check for question marks in the 'body' column
    contains_question_mark = aggregated_df['body'].str.contains('\?', na=False)
    
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
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()

# Example usage
# Assuming 'df' is your DataFrame and it has a 'body' column
# plot_percentage_of_questions(df, 'Percentage of Posts with Question Marks')


def plot_question_sentences_per_post(df_list, type_value, title, pdf=None):
    """
    Plot the distribution of the number of question sentences per post, replacing sequences of 
    multiple question marks with a single question mark before counting. This version aggregates
    data from multiple DataFrames based on the type_value.
    """
    plt.figure(figsize=(10, 6))

    # Define a function to count question sentences in a text after replacing "???" or "??" with "?"
    def count_question_sentences(text):
        if pd.notnull(text):
            text_modified = text.replace('???', '?').replace('??', '?')
            return text_modified.count('?')
        return 0

    # Merge DataFrames if necessary
    if len(df_list) > 1:
        aggregated_df = pd.concat(df_list, ignore_index=True)
    else:
        aggregated_df = df_list[0]

    # Apply the counting function to the appropriate text column
    text_column = 'body' if type_value == "user" else 'selftext'
    aggregated_df['question_sentences_count'] = aggregated_df[text_column].apply(count_question_sentences)

    # Plot the distribution of question sentence counts
    aggregated_df['question_sentences_count'].plot(kind='hist', bins=50, title=title)
    plt.xlabel('Number of Question Sentences per Post')
    plt.ylabel('Frequency')

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()

def plot_combined_distribution(user_df, family_df_list, column_name, title, xlabel, ylabel, bins=50, xlim=None, pdf=None):
    """
    Plot distributions for a given column from user posts and aggregated family posts 
    (submissions and replies) on the same graph for comparison.
    
    Parameters:
    - user_df: DataFrame for user data.
    - family_df_list: List of DataFrames for family data (submissions and replies).
    - column_name: The column name to plot the distribution for.
    - title: The title of the plot.
    - xlabel: The label for the X-axis.
    - ylabel: The label for the Y-axis.
    - bins: The number of bins to use for the histogram.
    - xlim: The limits for the X-axis.
    - pdf: Optional PdfPages object to save the plot, if provided.
    """
    plt.figure(figsize=(12, 8))
    
    # Aggregate family DataFrames
    if len(family_df_list) > 1:
        family_df = pd.concat(family_df_list, ignore_index=True)
    else:
        family_df = family_df_list[0]
    
    # Prepare data
    user_data = user_df[column_name].dropna()
    family_data = family_df[column_name].dropna()

    # Define bin edges for consistent binning across both datasets
    combined_data = pd.concat([user_data, family_data])
    if xlim:
        bin_edges = np.linspace(*xlim, bins)
    else:
        bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins)
    
    # Plot histograms
    plt.hist(user_data, bins=bin_edges, alpha=0.5, label='User Posts', edgecolor='black')
    plt.hist(family_data, bins=bin_edges, alpha=0.5, label='Family Posts (Submissions & Replies)', edgecolor='black')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    
    if xlim:
        plt.xlim(xlim)

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()
        plt.figure(figsize=(12, 8))
    
    # Aggregate family DataFrames
    if len(family_df_list) > 1:
        family_df = pd.concat(family_df_list, ignore_index=True)
    else:
        family_df = family_df_list[0]
    

    # Calculate kernel density estimates for user and family data
    x_range = np.linspace(min(user_data.min(), family_data.min()), max(user_data.max(), family_data.max()), 1000)
    kde_user = gaussian_kde(user_data)(x_range)
    kde_family = gaussian_kde(family_data)(x_range)
    
    # Plotting the KDEs
    plt.plot(x_range, kde_user, label='User Posts', color='blue')
    plt.fill_between(x_range, kde_user, color='blue', alpha=0.5)
    plt.plot(x_range, kde_family, label='Family Posts (Submissions & Replies)', color='red')
    plt.fill_between(x_range, kde_family, color='red', alpha=0.5)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    
    if xlim:
        plt.xlim(xlim)

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()
    else:
        plt.show()  # Display the plot
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
def extract_text_lengths(entry):
    """
    Recursively extracts text lengths from submissions and comments.
    - For submissions, uses 'selftext'
    - For comments, uses 'body'
    Entries with 'replies' will have their replies processed similarly.
    """
    text_lengths = []
    family_lengths=[]
    # Check if the entry is a submission or a commen
    if not isinstance(entry, dict):
        return [],[]
    # Your previous logic here with a minor adjustment
    if entry.get('type') == 'submission':
        text_1 = entry.get('title', '')
        text_2 = entry.get('selftext', '')
        text_lengths.append(len(text_1.split()) + len(text_2.split()))
    elif entry.get('type') == 'comment':  # Assuming the only other type is 'comment'
        text = entry.get('body', '')
        text_lengths.append(len(text.split()))

    # Process replies if they exist and are not NaN
    if 'replies' in entry and isinstance(entry['replies'], list):
        for reply in entry['replies']:
            text_length, _ = extract_text_lengths(reply)
            family_lengths.extend(text_length)
    if 'parent' in entry and isinstance(entry['parent'], dict):
        if(entry['parent']['type']=="comment"):
            par = entry['parent']
            text_length, _ = extract_text_lengths(par)
            family_lengths.extend(text_length)
    if 'submission' in entry and isinstance(entry['submission'], dict):
        subm = entry['submission']
        text_length, _ = extract_text_lengths(subm)
        family_lengths.extend(text_length)
    return text_lengths, family_lengths
def plot_text_length(text_lengths, family_lengths, pdf=None):
    # Filter out non-positive values before log transformation
    text_lengths_log = np.log([length for length in text_lengths if length > 0])
    family_lengths_log = np.log([length for length in family_lengths if length > 0])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(text_lengths_log, bw_adjust=0.5, label='Users', fill=True)
    sns.kdeplot(family_lengths_log, bw_adjust=0.5, label='Family', fill=True)
    plt.title('Log-transformed Distribution of Text Lengths: Family vs. Users')
    plt.xlabel('Log-transformed Text Length, 10^x')
    plt.ylabel('Density')
    plt.legend()
    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
    else:
        plt.show()
    x_range = np.linspace(min(min(text_lengths), min(family_lengths)), 
                          300, 1000)
    
    # Calculate kernel density estimates for each list
    kde_text = gaussian_kde(text_lengths, bw_method=0.5)(x_range)
    kde_family = gaussian_kde(family_lengths, bw_method=0.5)(x_range)

    plt.figure(figsize=(10, 6))
    
    # Plotting the kernel density estimate for "Users"
    plt.plot(x_range, kde_text, label='Users', color='blue')
    plt.fill_between(x_range, kde_text, color='blue', alpha=0.5)
    
    # Plotting the kernel density estimate for "Family"
    plt.plot(x_range, kde_family, label='Family', color='red')
    plt.fill_between(x_range, kde_family, color='red', alpha=0.5)
    
    plt.title('Distribution of Text Lengths: Family vs. Users')
    plt.xlabel('Text Length')
    plt.ylabel('Density')
    plt.legend()

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()  # Close the plot to avoid display issues
    else:
        plt.show()  # Display the plot

def plot_upvotes(user_upvotes, family_upvotes, pdf=None):
    # Filter out non-positive values before log transformation
    plt.figure(figsize=(10, 6))
    sns.kdeplot(user_upvotes, bw_adjust=0.5, label='Users', fill=True)
    sns.kdeplot(family_upvotes, bw_adjust=0.5, label='Family', fill=True)

    # Set the x-axis limits to -1 to 1
    plt.xlim(0, 1)

    plt.title('Distribution of Upvotes: Family vs. Users')
    plt.xlabel('Upvotes')
    plt.ylabel('Density')
    plt.legend()
    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
    else:
        plt.show()
    x_range = np.linspace(0, 
                          1, 1000)
    
    # Calculate kernel density estimates for each list
    kde_text = gaussian_kde(user_upvotes, bw_method=0.5)(x_range)
    kde_family = gaussian_kde(family_upvotes, bw_method=0.5)(x_range)

    plt.figure(figsize=(10, 6))
    
    # Plotting the kernel density estimate for "Users"
    plt.plot(x_range, kde_text, label='Users', color='blue')
    plt.fill_between(x_range, kde_text, color='blue', alpha=0.5)
    
    # Plotting the kernel density estimate for "Family"
    plt.plot(x_range, kde_family, label='Family', color='red')
    plt.fill_between(x_range, kde_family, color='red', alpha=0.5)
    
    plt.title('Distribution of Upvotes: Family vs. Users')
    plt.xlabel('Upvotes')
    plt.ylabel('Density')
    plt.legend()

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()  # Close the plot to avoid display issues
    else:
        plt.show()  # Display the plot
def plot_scores_tuple(all_upv_down_user, all_upv_down_fam, pdf=None):
    filtered_upv_down_user = [tup for tup in all_upv_down_user if None not in tup]
    filtered_upv_down_fam = [tup for tup in all_upv_down_fam if None not in tup]

    # Separate the tuples into individual lists
    user_scores_x, user_scores_y = zip(*filtered_upv_down_user) if filtered_upv_down_user else ([], [])
    family_scores_x, family_scores_y = zip(*filtered_upv_down_fam) if filtered_upv_down_fam else ([], [])
    user_scores_x = np.array(user_scores_x)
    user_scores_y = np.array(user_scores_y)
    family_scores_x = np.array(family_scores_x)
    family_scores_y = np.array(family_scores_y)
    user_scores_log_x = np.log([length for length in user_scores_x if length > 0])
    user_scores_log_y = np.log([length for length in user_scores_y if length > 0])
    family_scores_log_x = np.log([length for length in family_scores_x if length > 0])
    family_scores_log_y = np.log([length for length in family_scores_y if length > 0])
    # Convert to numpy arrays for easier manipulation


    # Plot 1: Distribution of Scores for User X values
    plt.figure(figsize=(10, 6))
    sns.kdeplot(user_scores_log_x, bw_adjust=0.5, label='Users upvotes', fill=True)
    sns.kdeplot(user_scores_log_y, bw_adjust=0.5, label='Users downvotes', fill=True)
    sns.kdeplot(family_scores_log_x, bw_adjust=0.5, label='Family upvotes', fill=True)
    sns.kdeplot(family_scores_log_y, bw_adjust=0.5, label='Family downvotes', fill=True)
    plt.title('Distribution of Scores: Family vs. Users')
    plt.xlabel('Log-transformed Scores, 10^x')
    plt.ylabel('Density')
    plt.legend()
    if pdf:
        pdf.savefig()
    plt.close()

    # Plot 2: Distribution of Scores for User Y values
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1000)
    sns.kdeplot(user_scores_x, bw_adjust=0.5, label='Users Up', fill=True)
    sns.kdeplot(family_scores_x, bw_adjust=0.5, label='Family Up', fill=True)
    sns.kdeplot(user_scores_y, bw_adjust=0.5, label='Users Down', fill=True)
    sns.kdeplot(family_scores_y, bw_adjust=0.5, label='Family Down', fill=True)
    plt.title('Distribution of Y Scores: Family vs. Users')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.legend()
    if pdf:
        pdf.savefig()
    plt.close()
    x_range = np.linspace(0, 
                          5000, 1000)
    
    # Calculate kernel density estimates for each list
    kde_user_x= gaussian_kde(user_scores_x, bw_method=0.5)(x_range)
    kde_user_y= gaussian_kde(user_scores_y, bw_method=0.5)(x_range)
    kde_family_x = gaussian_kde(family_scores_x, bw_method=0.5)(x_range)
    kde_family_y = gaussian_kde(family_scores_y, bw_method=0.5)(x_range)
    plt.figure(figsize=(10, 6))
    
    # Plotting the kernel density estimates
    plt.plot(x_range, kde_user_x, label='User Up', color='blue')
    plt.fill_between(x_range, kde_user_x, color='blue', alpha=0.3)
    
    plt.plot(x_range, kde_user_y, label='User Down', color='cyan')
    plt.fill_between(x_range, kde_user_y, color='cyan', alpha=0.3)
    
    plt.plot(x_range, kde_family_x, label='Family Up', color='red')
    plt.fill_between(x_range, kde_family_x, color='red', alpha=0.3)
    
    plt.plot(x_range, kde_family_y, label='Family Down', color='orange')
    plt.fill_between(x_range, kde_family_y, color='orange', alpha=0.3)
    
    plt.title('Distribution of Scores: Family vs. Users')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()  # Close the plot to avoid display issues
    else:
        plt.show()  # Display the plot

def plot_scores(user_scores, family_scores, pdf=None):
    # Filter out non-positive values before log transformation
    user_scores_log = np.log([length for length in user_scores if length > 0])
    family_scores_log = np.log([length for length in family_scores if length > 0])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(user_scores_log, bw_adjust=0.5, label='Users', fill=True)
    sns.kdeplot(family_scores_log, bw_adjust=0.5, label='Family', fill=True)
    plt.title('Log-transformed Distribution of Scores: Family vs. Users')
    plt.xlabel('Log-transformed Scores, 10^x')
    plt.ylabel('Density')
    plt.legend()
    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
    else:
        plt.show()
    x_range = np.linspace(min(min(user_scores), min(family_scores)), 
                          5000, 1000)
    
    # Calculate kernel density estimates for each list
    kde_text = gaussian_kde(user_scores, bw_method=0.5)(x_range)
    kde_family = gaussian_kde(family_scores, bw_method=0.5)(x_range)

    plt.figure(figsize=(10, 6))
    
    # Plotting the kernel density estimate for "Users"
    plt.plot(x_range, kde_text, label='Users', color='blue')
    plt.fill_between(x_range, kde_text, color='blue', alpha=0.5)
    
    # Plotting the kernel density estimate for "Family"
    plt.plot(x_range, kde_family, label='Family', color='red')
    plt.fill_between(x_range, kde_family, color='red', alpha=0.5)
    
    plt.title('Distribution of Scores: Family vs. Users')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()  # Close the plot to avoid display issues
    else:
        plt.show()  # Display the plot
def plot_question_counts(all_user_questions, all_family_questions, pdf=None):
    # Calculate the kernel density estimate for each category
    x_range = np.linspace(min(all_user_questions + all_family_questions), 
                         7, 1000)
    kde_user = gaussian_kde(all_user_questions, bw_method=0.5)(x_range)
    kde_family = gaussian_kde(all_family_questions, bw_method=0.5)(x_range)

    # Calculate the stacked densities
    cumulative_kde = np.add(kde_user, kde_family)

    plt.figure(figsize=(10, 6))

    # Plot the cumulative density as the background area
    plt.fill_between(x_range, cumulative_kde, color="lightgray", label='Cumulative')
    
    # Plot the individual densities on top
    plt.fill_between(x_range, kde_user, color="blue", alpha=0.5, label='Users')
    plt.fill_between(x_range, kde_user, kde_user + kde_family, color="red", alpha=0.5, label='Family')

    plt.title('Stacked Distribution of Number of Questions: Family vs. Users')
    plt.xlabel('Question Count')
    plt.ylabel('Density')
    plt.legend()

    if pdf:
        pdf.savefig()  # Save to PDF if a PdfPages object is provided
        plt.close()  # Close the plot to avoid display issues
    else:
        plt.show()  # Display the plot
def extract_question_counts(entry):
    """
    Recursively extracts the number of questions from submissions and comments,
    treating consecutive question marks as a single question.
    - For submissions, analyzes 'title' and 'selftext'.
    - For comments, analyzes 'body'.
    Entries with 'replies' will have their replies processed similarly.
    """
    questions_counts = []
    family_questions = []

    # Function to replace consecutive question marks with a single one
    def normalize_questions(text):
        return re.sub(r'\?+', '?', text)

    if not isinstance(entry, dict):
        return [], []

    if entry.get('type') == 'submission':
        title_questions = normalize_questions(entry.get('title', '')).count('?')
        selftext_questions = normalize_questions(entry.get('selftext', '')).count('?')
        questions_counts.append(title_questions + selftext_questions)
    elif entry.get('type') == 'comment':  # Assuming the only other type is 'comment'
        body_questions = normalize_questions(entry.get('body', '')).count('?')
        questions_counts.append(body_questions)

    # Process replies if they exist
    if 'replies' in entry and isinstance(entry['replies'], list):
        for reply in entry['replies']:
            reply_questions, family_reply_questions = extract_question_counts(reply)
            questions_counts.extend(reply_questions)
            family_questions.extend(family_reply_questions)
    if 'parent' in entry and isinstance(entry['parent'], dict):
        if(entry['parent']['type']=="comment"):
            par = entry['parent']
            subm_questions, family_subm_questions = extract_question_counts(par)
            family_questions.extend(subm_questions)
    if 'submission' in entry and isinstance(entry['submission'], dict):
        subm = entry['submission']
        subm_questions, family_subm_questions = extract_question_counts(subm)
        family_questions.extend(subm_questions)

    return questions_counts, family_questions
def extract_scores_and_upvotes(entry):
    """
    Recursively extracts 'score' and 'upvote_ratio' from submissions and comments.
    Entries with 'replies' will have their data processed similarly.
    """
    scores_user = []
    scores_fam = []
    upvote_user=[]
    upvote_fam = []
    upvotes_downvotes_user=[]
    upvotes_downvotes_fam=[]
    # Check if the entry is a submission or a comment
    if not isinstance(entry, dict):
        return [], [],[],[]
    
    # Extract 'score' and 'upvote_ratio' where applicable
    if entry.get('type') == 'submission':
        scores_user.append(entry.get('score', 0))
        # 'upvote_ratio' might not be present in all data structures (e.g., comments), so we check its existence
        upvote_user.append(entry.get('upvote_ratio', 0.0))
        upvotes_downvotes_user.append(calculate_votes(entry.get('score',0),entry.get('upvote_ratio',0.0)))
    elif entry.get('type') == 'comment':
        scores_user.append(entry.get('score', 0))
        # Comments may not have 'upvote_ratio', so it's not extracted here
    
    # Process replies if they exist
    if 'replies' in entry and isinstance(entry['replies'], list):
        for reply in entry['replies']:
            scores_user_repl,scores_fam_repl, upvote_user_repl, upvote_fam_repl,upvotes_downvotes_user_repl, upvotes_downvotes_fam_repl = extract_scores_and_upvotes(reply)
            scores_fam.extend(scores_user_repl)
            upvote_fam.extend(upvote_user_repl)
    
    # Process 'parent' and 'submission' if they exist and are dictionary objects
    if 'parent' in entry and isinstance(entry['parent'], dict) and entry['parent'].get('type') == "comment":
        scores_user_par,scores_fam_par, upvote_user_par, upvote_fam_par, upvotes_downvotes_user_par, upvotes_downvotes_fam_par = extract_scores_and_upvotes(entry['parent'])
        scores_fam.extend(scores_user_par)
        upvote_fam.extend(upvote_user_par)
        
    if 'submission' in entry and isinstance(entry['submission'], dict):
        scores_user_subm,scores_fam_subm, upvote_user_subm, upvote_fam_subm, upvotes_downvotes_user_subm, upvotes_downvotes_fam_subm = extract_scores_and_upvotes(entry['submission'])
        scores_fam.extend(scores_user_subm)
        upvote_fam.extend(upvote_user_subm)
        upvotes_downvotes_fam.extend(upvotes_downvotes_user_subm)
    return scores_user,scores_fam, upvote_user, upvote_fam, upvotes_downvotes_user, upvotes_downvotes_fam
def calculate_votes(score, upvote_ratio):
    """
    Calculate the number of upvotes and downvotes based on score and upvote ratio.
    
    Args:
    - score (int): The score of the post (upvotes - downvotes).
    - upvote_ratio (float): The ratio of upvotes to total votes.
    
    Returns:
    - tuple: A tuple containing the number of upvotes and downvotes (upvotes, downvotes).
             Returns (None, None) if it's impossible to determine absolute values.
    """
    if upvote_ratio == 1:  # All votes are upvotes
        upvotes = score
        downvotes = 0
    elif upvote_ratio == 0:  # Edge case scenario, assuming no votes if ratio is 0
        upvotes = 0
        downvotes = 0
    elif upvote_ratio == 0.5:  # Equal number of upvotes and downvotes, but exact numbers are indeterminate
        return None, None  # Cannot determine absolute values from score and upvote_ratio alone
    else:
        # Calculate upvotes using the derived formula
        upvotes = int(round(score / (2 * upvote_ratio - 1)))
        # Calculate downvotes based on the score and upvotes
        downvotes = upvotes - score

    return (upvotes, downvotes)

@click.command()
@click.option("--json-path", type=click.Path(dir_okay=False, exists=True), help="Path to the JSON Lines file.")
def main(json_path):
    print("----------------------")
    data=[]
    with open(json_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert the list of JSON objects into a DataFrame
    df_1 = pd.DataFrame(data)
    df = load_jsonl_to_dataframe(json_path)
    replies_df= child_replies_df(df) #non user replies
    submission_df = parent_submission_df(df) #non user submissions
    submission_df['body'] = submission_df['selftext']
    print(submission_df.columns)
    with PdfPages('data_plots_json.pdf') as pdf:

        plot_posts_overview(df, pdf)
        # 2
        plot_activity_over_time(df,  pdf)
        plot_activity_over_time(df,  pdf)

        all_text_lengths = []
        all_family_text_lengths = []
        all_family_questions=[]
        all_user_questions=[]
        all_upvotes_user=[]
        all_scores_user=[]
        all_upvotes_fam=[]
        all_scores_fam=[]
        all_upv_down_user=[]
        all_upv_down_fam=[]
        for index,row in df_1.iterrows():
            row_dict = row.to_dict()
            text_lengths, family_lengths=extract_text_lengths(row_dict)
            questions_counts, family_questions = extract_question_counts(row_dict)
            scores_user, scores_fam, upvote_user, upvote_fam, upv_down_u, upv_down_fam= extract_scores_and_upvotes(row_dict)
            all_family_questions.extend(family_questions)
            all_user_questions.extend(questions_counts)
            all_upvotes_user.extend(upvote_user)
            all_upvotes_fam.extend(upvote_fam)
            all_scores_fam.extend(scores_fam)
            all_scores_user.extend(scores_user)
            all_upv_down_user.extend(upv_down_u)
            all_upv_down_fam.extend(upv_down_fam)
            all_text_lengths.extend(length for length in text_lengths if length > 0)
            all_family_text_lengths.extend(length for length in family_lengths if length > 0)

        plot_text_length(all_text_lengths, all_family_text_lengths,pdf)
        plot_question_counts(all_user_questions, all_family_questions,pdf)

        plot_scores(all_scores_user, all_scores_fam, pdf)
        plot_upvotes(all_upvotes_user, all_upvotes_fam, pdf)
        
        plot_scores_tuple(all_upv_down_user, all_upv_down_fam, pdf)
        # Add more plots or analysis as required

if __name__ == '__main__':
    main()

