* Encoding: UTF-8.
* September 2025
* Frederic Malharin
* dataset : https://www.kaggle.com/datasets/bhavikjikadara/chatgpt-user-feedback?resource=download


* -------------------- STEP 1 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* IMPORT CSV INTO SPSS.

GET DATA 
  /TYPE=TXT
  /FILE="C:\PATH\clean_chatgpt_reviews.csv"
  /DELCASE=LINE
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
    id F8.0
    userName A100
    content A1000
    score F8.0
    thumbsUpCount F8.0
    at A20.
CACHE.
EXECUTE.

* -------------------- STEP 2 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    EXPLORATION AND RESHAPING


* -------------------- STEP 2.1 --------------------
* Renaming the columns

DISPLAY DICTIONARY.

RENAME VARIABLES (content = review).
EXECUTE.

RENAME VARIABLES (at = dateTime).
EXECUTE.

DISPLAY DICTIONARY.

* -------------------- STEP 2.2 --------------------
* Add labels to original variables.

VARIABLE LABELS id "Original id variable"
                userName "Original userName variable"
                review "Original review variable"
                score "Original score variable"
                thumbsUpCount "Original thumbsUpCount variable"
                dateTime "Original dateTime variable".
EXECUTE.

* -------------------- STEP 2.3 --------------------
* Convert dateTime to SPSS datetime.

* FORMATS dateTime (DATETIME20).
* DOES NOT WORK.

* Create a new datetime variable from the string.
COMPUTE dateTime_num = NUMBER(dateTime, DATETIME20).
FORMATS dateTime_num (DATETIME20).
VARIABLE LABELS dateTime_num "Converted datetime variable".
EXECUTE.

* -------------------- STEP  2.4 --------------------
Creating new columns for date, time, year, month, day, day of the week, yearMonth.

* Extract just the calendar date (no time).
COMPUTE date = XDATE.DATE(dateTime_num).
FORMATS date (DATE11).
VARIABLE LABELS date "Review date".

* Extract just the clock time.
COMPUTE time = XDATE.TIME(dateTime_num).
FORMATS time (TIME8).
VARIABLE LABELS time "Review time".

* Extract year, month, day, weekDay.
COMPUTE year = XDATE.YEAR(dateTime_num).
COMPUTE month = XDATE.MONTH(dateTime_num).
COMPUTE day = XDATE.MDAY(dateTime_num).
COMPUTE weekDay = XDATE.WKDAY(dateTime_num).

* Creation of yearMonth.
STRING yearMonth (A7).
COMPUTE yearMonth = CONCAT(
    STRING(year, F4.0), "-", 
    STRING(month, F2.0)).
EXECUTE.


* Edit labels for year, month, day,weekDay, yearMonth.
VARIABLE LABELS year "Year of review"
                month "Month of review"
                day "Day of month of review"
                weekDay "Day of week of review"
                yearMonth "Year and month combined".
EXECUTE.


* -------------------- STEP  2.5 --------------------
Create a user duplicate variable. These lines of code create 2 variables that flag if a user name appears for the first time and last time.
* If a user name has a 1 for bothvariables, it is unique. If not, it has several entries.

SORT CASES BY userName.
MATCH FILES /FILE=* /BY userName /FIRST=first /LAST=last.
SELECT IF NOT(first=1 AND last=1).
COMPUTE isDuplicate = 0.
IF (NOT(first=1 AND last=1)) isDuplicate = 1.
VARIABLE LABELS isDuplicate "Flag for duplicate users (1=duplicate, 0=unique)".
VALUE LABELS isDuplicate 0 "Unique user" 1 "Duplicate user".
EXECUTE.

DELETE VARIABLES first last.
EXECUTE.



* -------------------- STEP  3 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* EXPLORATORY DATA ANALYSIS


* -------------------- STEP  3.1 --------------------
* Checking for duplicates.

FREQUENCIES VARIABLES=isDuplicate.
* We have 16932 rows of data in the dataset.
* 100% of the data in this dataset has a userName that is duplicated, but without being fully duplicated, with different reviews and dateTime. That's a bit strange to say the least.



* -------------------- STEP  3.2 --------------------
* Time variables.

FREQUENCIES VARIABLES=year.
* 75% of the data has a year of 2024, 25% has 2023.

FREQUENCIES VARIABLES=yearMonth.
* 35% of the data comes from the month of May 2024 alone.

FREQUENCIES VARIABLES=day.
* Day 12 and day 17 stand out with 10.7% and 14.1% of the entries. There may have been some specific days with lots of entries or duplicates.

FREQUENCIES VARIABLES=weekDay.
* NOTE: 1 is Sunday, 7 is Saturday.
* There are about twice as many entries on Friday and Sunday than the other days (19% and 22% vs 10 to 14% for the other day). I wont explore more this topic.

* Dive into the days 12 and 17 with a filter.
USE ALL.
COMPUTE filter_$ = (day=12 OR day=17).
FILTER BY filter_$.
EXECUTE.

FREQUENCIES VARIABLES=yearMonth.
* 84% of the data for those 2 days of the month come from the month of May 2024. Those 2 days represent 60% of the entries for May 2024 (3520 out of 5894 rows of data)
* I guess that something special happened on those days 12th and 17th of May 2024.
* A quick check at the calendar : those days were a Sunday and a Friday. This explains the higher amount of data for Sundays and Fridays.
* A quick visual inspection of the data allowed me to see that most of the data for the 17th of May 2024 is triplicated. I will deduplicate the data.

* Removing the filter and deleting the filter variable.
FILTER OFF.
USE ALL.
EXECUTE.
DELETE VARIABLES filter_$.

* -------------------- STEP  3.3 --------------------
* Deduplicating the data.

SORT CASES BY userName review score dateTime.

* Creating a helper variable to flag the duplicates. 1 for the first occurrence of a comination of userName review score dateTime, then 0 for all the other occurrences.
MATCH FILES
  /FILE=*
  /BY userName review score dateTime
  /FIRST=FirstCase.
EXECUTE.

* Dropping the duplicates by keeping only the first occurrence.
SELECT IF FirstCase=1.
EXECUTE.

* Dropping the helper variable.
DELETE VARIABLES FirstCase.

* Checking for duplicates.
FREQUENCIES VARIABLES=isDuplicate.
* Now we still have 100% of duplicated userNames, but we have dropped about 2500 rows of data and have only 14421 rows left.
* We have dropped the true duplicates, those who have the same userName, review, score and dateTime.



* Rechecking the frequencies for our time variables.
FREQUENCIES VARIABLES=year.
FREQUENCIES VARIABLES=yearMonth.
FREQUENCIES VARIABLES=day.
FREQUENCIES VARIABLES=weekDay.
* The excess of data on Sunday 12th and Friday 17th of May 2024 have now disappeared.
* May 2024 is still the month with the highest amount of data, but not as much as before 23.5% instead of 35%.




* -------------------- STEP  3.4 --------------------
*Variable  thumbsUpCount.

DESCRIPTIVES VARIABLES=thumbsUpCount
  /STATISTICS=MEAN STDDEV MIN MAX.
* The variable thumbsUpCount has mainly a mean of 0.86 and a standard deviation of 17.637, a lot higher than the mean, which suggest some very skewed data. Let's look at it in detail.
  
* Frequencies to check distribution.
FREQUENCIES VARIABLES=thumbsUpCount
  /STATISTICS=MINIMUM MAXIMUM MEAN MEDIAN
  /PERCENTILES=25 50 75 90 95 99
  /ORDER=ANALYSIS.

* Out of the 14421 rows of data, 95.5% of the values for the variable thumbsUpCount are 0, 2.4% are 1, and 2.1% are integers between 2 and 1017.
* Some people have put a lot of thumbs up, while most people haven't put any.
* We could check the correlation between the thumbs up and the score.


* -------------------- STEP  3.5 --------------------
* Variable score.

FREQUENCIES VARIABLES=score
  /BARCHART FREQ
  /PIECHART FREQ.

* 72.8% of the entries have a score of 5 12.4% of 4, 4.8% of 3, 2.2% of 1, and 7.7% of 1 


* -------------------- STEP  3.6 --------------------
* Correlation between thumbsUpCount and score.

* Create binary version: 0 = no thumbs up, 1 = at least one.
RECODE thumbsUpCount (0=0) (ELSE=1) INTO hasThumbsUp.
VARIABLE LABELS hasThumbsUp "Has at least 1 thumbs up (0=no, 1=yes)".
VALUE LABELS hasThumbsUp 0 "No thumbs up" 1 "Has thumbs up".
EXECUTE.


MEANS TABLES=score BY hasThumbsUp
  /CELLS=MEAN COUNT STDDEV.
* The mean of the score is higher when there is no thumbs up (4.46) than when there is thumbs up (3.27).
* This is strange. I would have expected higher scores when the user has given a thumb up.

CORRELATIONS
  /VARIABLES=score hasThumbsUp
  /PRINT=TWOTAIL NOSIG.
* The Pearson coefficient is -0.209. There is a weak negative correlation between the 2 variables.
* This confirms the previous observation, the score is lower when there are thumbs up.

* Let's move on.




* -------------------- STEP  4 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* TIME TRENDS.

* -------------------- STEP  4.1 --------------------
* Plotting score vs yearMonth

* Creating a date variable with the first day of the month as a true SPSS date. this avoids plotting issues with a date as x axis.
COMPUTE monthDate = DATE.MDY(month, 1, year).
FORMATS monthDate (DATE11).
VARIABLE LABELS monthDate "First day of month".
EXECUTE.

* Plot mean score over time using the date axis.
GGRAPH
  /GRAPHDATASET NAME="g" VARIABLES=monthDate score
  /GRAPHSPEC SOURCE=INLINE.
BEGIN GPL
  SOURCE: s = userSource(id("g"))
  DATA: monthDate = col(source(s), name("monthDate"))
  DATA: score     = col(source(s), name("score"))
  GUIDE: axis(dim(1), label("Month"))
  GUIDE: axis(dim(2), label("Average score"))
  ELEMENT: line(position(summary.mean(monthDate*score)))
END GPL.


* On the graph, the mean score for June 2024 seem lower than the average. but it could be just due to the scale. Let's run a t test.

* -------------------- STEP  4.2 --------------------
* Statistics of score vs yearMonth

MEANS TABLES=score BY yearMonth
  /CELLS=MEAN COUNT.
* The mean for the dataset is 4.40.

* Select only the specific month.
TEMPORARY.
SELECT IF (yearMonth = "2024- 6").

* One-sample t-test: compare month’s scores to population mean (4.40).
T-TEST
  /TESTVAL=4.40
  /VARIABLES=score.

* SPSS tests both the one sided p value and the two sided p value.
* Two sided p value : Is June’s mean score different from 4.40, either higher or lower?
* One sided p value : Is June’s mean score specifically lower than 4.40?

* The p value is < 0.001, both one sided and two sided. The probability of seeing this difference just by chance is less than 0.1%.
* The mean score for June is lower than the review meanscore, it is unlikely that this difference is due to chance.
* What could be the reason for this difference ?
* I won't explore it further
 

* -------------------- STEP  4.3 --------------------
* TEXT ANALYSIS 
* I downloaded the text analysis bundle from the extension hub.

* I didn't find how to use Syntax to perform the text analysis.
* Analyze → Descriptive Statistics → Text Analysis
    * Variables tab → variable : review
    * Statistics  tab → Calculate frequencies with max items = 30 (no stem words)
    * OK

* The most common word combinations are good app, best app, nice app, great app, but also please try later (35 occrruences).

* I will run the same analysis on the data with scores < 4.
TEMPORARY.
SELECT IF (score < 4).
* Analyze → Descriptive Statistics → Text Analysis
    * Variables tab → variable : review
    * Statistics  tab → Calculate frequencies with max items = 30 (no stem words)
    * OK

* There is now a lot less positive words and word combinations. "please try later" is now leading the 3 words combinations with 27 occurrences.



* I tried to analyze the sentiments, but the SPSS froze every time.
* Analyze → Descriptive Statistics → Text Analysis
    * Variables tab → variable : review
    * Statistics  tab → Calculate sentiments
    * negative, positive, neutral
    * OK
