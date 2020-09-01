# **Bike sharing system**

Data source : https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset 

We are going to perform analysis on this dataset which contains the hourly count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information. Dataset consists of 17379 rows and 17 features (columns).

## **Abstract**

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues.


This readme describes how the analysis is perfomed on this bike sharing systems dataset.

## **Objective** 

Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by these systems make them attractive for the research. Opposed to other transport services such as bus or subway, the duration of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system into a virtual sensor network that can be used for sensing mobility in the city. Hence, it is expected that most of important events in the city could be detected via monitoring these data.

## **Process Outline** 

<img src = "https://github.com/pinkesh-nayak/job_aggregator/blob/master/data/process.PNG">


## **Data Source and Access Rights**<br>
### **Randstad: Web Scraping**<br>
Link: https://www.randstad.com/jobs/united-states/q-data-science/<br>
1. Check robot.txt of Randstad | It allows scraping with crawl delay of 5 seconds. <br>

### **Adzuna API**<br>
Link: https://developer.adzuna.com/overview <br>
We need to create a developer account to get an app_key and aap_id to access the API.<br>
Quering the API : https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={YOUR_APP_ID}&app_key={YOUR_APP_KEY}<br>
**Important Terms and Conditions** <br>
1. There are no rate limits present.<br>
Detailed verion of the terms and conditions : https://www.adzuna.co.uk/terms-and-conditions.html<br>
