
# coding: utf-8

# In[189]:


print("Hello world I am in a Jupyter Notebook!")


# Ok, we'll start the notebook now. Taking a sports oriented approach on my first notebook because I'm getting more and more impatient for basketball season to begin.

# In[190]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
#Reading in the datasets of NBA player's contract salaries and stats from the past season
df_salaries = pd.read_csv('Attempt2_salaries.csv')
df_stats = pd.read_csv('Attempt2_stats.csv')


# In[191]:


df_salaries.tail()


# In[192]:


df_salaries.head()
#just checking out the data here to see what a dataframe looks like


# In[193]:


df_stats.head()


# In[194]:


df_stats.tail()


# Have to replace the dollar signs in the salaries data set for the information to be useful. 

# In[195]:


#Removing all $ signs from salary data set
#this took me so long to get right
df_salaries['2018-19'] = [x.strip('$') for x in df_salaries['2018-19']]
df_salaries.head()
#df_salaries['2019-20'].str.replace('$', '')


# The other columns of salaries (not 2018-19) have Nan values and are also unimportant to this analysis; I will remove those columns to clean this data set more. 

# In[196]:


df_salaries = df_salaries.drop(['2019-20'], axis=1)
df_salaries = df_salaries.drop(['2020-21'], axis=1)
df_salaries.head()


# In[202]:


#Converting the 2018-19 salaries type from 'object' to 'float64' 
df_salaries[['2018-19']] = df_salaries[['2018-19']].apply(pd.to_numeric)


# Both these data sets originate from Basketball Reference. Basketball Reference does a good job of having complete, clean data sets for the most part. I'm seeing some repeats (?) in the stats data set so something to look more into.

# Now I will do some exploratory analysis to see what can be accomplished by combining these two data sets. Since there are many statistics it'll be interesting to see what ones have some (if any) correlation with salaries.

# In[203]:


#to suppress scientific notation in the description
pd.options.display.float_format = '{:.0f}'.format
df_salaries['2018-19'].describe(percentiles=[.20, .50, .70, .90])


# In[204]:


#TOV = Turnovers per game and PS/G = points per game
sns.lmplot('TOV', 'PS/G', data=df_stats, fit_reg=False)


# On its own, the salary data set does not provide much value as far as exploration goes. It needs to be combined with another stat to formulate some quantitative analysis. It would be more interesting to see how the salaries compare to a numeric value like:
# 1. Salary vs. turnovers per game or salary vs. points per game
# 2. Possibly analysis on individual teams and seeing if players from specific teams have salaries that don't necessarily correlate to what they "should be getting" based on stats
# 3. Using the information from Player's stats to perform predictive analysis on salaries (for a later project). 

# Coming to a realization that the rows in the Player's Stats set are duplicated for players who were traded during the season, representing their stats from each team they played for. This can be approached in different ways:
# 1. Only having the row which contains the highest stats for that player
# 2. Averaging all the stats for that player and merging the rows into one
# 
# Not sure what the best approach is so for now I will keep all rows for all players.

# In[209]:


#merging the cleaned data sets
df_merged = pd.merge(df_salaries, df_stats, on='Player')
df_merged.head()


# Looks like the merge went well! All the repeat rows were given the correct same salary for however many times they were repeated which is good to see as well. Now on to some more informative visualizations.

# In[210]:


sns.lmplot('2018-19', 'TOV', data=df_merged, fit_reg=False)


# In[211]:


sns.lmplot('2018-19', 'PS/G', data=df_merged, fit_reg=False)


# It's really interesting to see that there are players who fall in a certain range for PPG but are getting paid a completely unexpected salary compared to others in that range (looking particularly in the top left of the dot plot above). These plots definitely give more insight into factors that can be affecting salaries.
# 
# It would be nice to be able to see which player was represented by each dot; interactiveness is another thing to look into for the future. 
