# import main library
import torch
import pandas as pd
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
# for scraping comment i will use selenium 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

token = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') # pre trained model i will use

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# lets just test

def your_review(text):
    token_res = token.encode(text,return_tensors='pt') # output of text will be torch format
    result = model(token_res)
    
    my_out = int(torch.argmax(result.logits[0]))  # out result from bad to good will be 0-4 marked
    
    return my_out

print(your_review("I love it"))


############# Sentiment Analysis of  M1 Ultra Mac Studio review ( Linus Tech Tips) 

option = webdriver.FirefoxOptions()
option.add_argument("--headless")
driver = webdriver.Firefox(options=option)

wait = WebDriverWait(driver,15)

url = "https://www.youtube.com/watch?v=8YjMIjLLIwA&t=49s"

driver.get(url) 

data = []

for item in range(40):
    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END) # we need get all tag names with body
    time.sleep(25)
    
# we are take only contet tag and add it our data file    
for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))): 
    data.append(comment.text)
    
    
    
# some cleaning we have to make 
z = data[0].split("\n")
my_data = []
for i in z:
    if len(i)>300:
        my_data.append(i)
    else:
        None

for com in data[1:]:
    my_data.append(com)
    
    
# then lets create our dataset ##################3
df = pd.DataFrame(my_data, columns=['comment']) ## randomly selected 100 comments


df.to_csv('out.csv')

df = pd.read_csv("out.csv")

df["result"] = df['comment'].apply(lambda x: your_review(x))

sns.histplot(data=df, x="result")
