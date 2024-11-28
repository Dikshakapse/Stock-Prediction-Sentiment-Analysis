import os
import praw
import json

# Check if the 'data' folder exists, if not, create it
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize Reddit instance with your credentials
reddit = praw.Reddit(
    client_id='1NmKO0sw_bpUfVndlPGgVA',           # Replace with your client ID
    client_secret='1w9gJR5p3xdpkuxXN5vcK5CYGoDcSg',   # Replace with your client secret
    user_agent='StockMovementAnalysis:v1.0:by /u/Far_Tadpole_816'  # Replace with your Reddit username
)

# Specify the subreddit you want to scrape
subreddit = reddit.subreddit('stocks')  # Change 'stocks' to the subreddit you want

# Scrape the top 10 posts from the subreddit
posts = []
for post in subreddit.top(limit=10):  # You can change the limit
    posts.append({
        'title': post.title,
        'score': post.score,
        'url': post.url,
        'comments': post.num_comments,
        'created': post.created_utc,
        'content': post.selftext
    })

# Save the scraped data to a JSON file in the 'data' folder
with open('data/raw_data.json', 'w') as outfile:
    json.dump(posts, outfile, indent=4)

print("Data scraped and saved to 'data/raw_data.json'")
