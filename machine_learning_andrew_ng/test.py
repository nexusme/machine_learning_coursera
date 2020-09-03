# import libraries for this question here

import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re
import requests
import urllib.request
import json
import numpy as np
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# from IPython.core.display import HTML


link = "https://www.yellowpages.com.au/search/listings?clue=Shopping+Centres&locationClue=sunnybank+hills+qld+4109&lat=&lon=&referredBy=www.yellowpages.com.au&selectedViewMode=list&eventType=sort&openNow=false&sortBy=relevance"
content = requests.get(link)
web_content = content.content
# mammal_data = json.loads(content.content)
