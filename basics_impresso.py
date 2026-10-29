# %%

#%pip install impresso
# %%

# Connecting to Impresso API
from impresso import connect, OR, AND, DateRange
client = connect(public_api_url="https://dev.impresso-project.ch/public-api/v1")

# %%

res = client.search.find(term="Pierre Landry", with_text_contents=True)
res

# %%
res = client.search.find(term="Peter Landry", with_text_contents=True)
res
# %%
# see cols
res.df.columns



# %%

# we will often move back and forth between the notebook and the Impresso App, 
# so a small function for constructing links is useful. 

# Impresso web app 
WEBAPP_URL = "https://dev.impresso-project.ch/app"

# Function to generate web app URLs for images (issue_mode=False) or for the article containing them (issue_mode=True)
def img_webapp_url(uid, issue_mode=True):
  if not issue_mode:
    return f"{WEBAPP_URL}/search/images/{uid}"
  issue_id, item_id = uid.split("-a-")
  return f"{WEBAPP_URL}/issue/{issue_id}-a/view?articleId={item_id}"

# %%

id = "depechetoulouse-1912-08-06-a-i0090"

url = img_webapp_url(id)
url


# %%



# %%

# find PASSIM cluster
 
from impresso import OR

client.search.find(text_reuse_cluster_id=OR("jdpl-1842-07-02-a-i0018"))
# %%
