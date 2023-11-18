
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import requests
import numpy as np
import json
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder

pd.set_option('display.max_columns', None)

# Takes game finder for nba_api and gameDate which should be the first game of the season
# Loops and finds each game in the season stopping once 1230 games have been found
# Puts all data into a dataframe
def getSeason (games, gameDate):

    # Get index for first game 2021-2022
    games.reset_index()
    iSeason = games[games.GAME_DATE.str[:] == gameDate].index[0]
    print(games.loc[iSeason, 'SEASON_ID'])

    seasonID = "2" + gameDate[0:4]
    seasonGames = games[(games.SEASON_ID == seasonID)].copy().reset_index()

    i = seasonGames[seasonGames.GAME_DATE.str[:] == gameDate].index[-1]
    print(i)

    # Create new dataframe to add all data to
    dataFrame = pd.DataFrame(columns=['Game Date', 'Home', 'Home Pts', 'Away', 'Away Pts', 'WL'])

    # Stop looping once we've found all games for the season
    while (i >= 0):
        home = ""
        away = ""
        homePts = 0
        awayPts = 0
        winLoss = ""

        # Since NBA_API creates a 2 entries per game (One for away, one for home) skip any entries for away teams
        checkDup = seasonGames.loc[i, 'MATCHUP']
        if (not(checkDup[4:6] == "vs")):
            i -= 1
            continue

        # Get the game ID
        gameID = seasonGames.loc[i, 'GAME_ID']

        # Pulls two rows, one for away stats one for home
        currentGame = games[(games.GAME_ID == gameID)].copy().reset_index()

        # Check if current game hasn't been played yet to stop loop
        if (currentGame.empty):
            print("Current Game EMPTY")
            break

        gameDate = currentGame.loc[0, 'GAME_DATE']

        # Get information from both home and away
        for j in range(len(currentGame)):
            homeAway = currentGame.loc[j, 'MATCHUP']
            if (homeAway[4:6] == "vs"):
                home = currentGame.loc[j, 'TEAM_NAME']
                homePts = currentGame.loc[j, 'PTS']
                winLoss = currentGame.loc[j, 'WL']
            else:
                away = currentGame.loc[j, 'TEAM_NAME']
                awayPts = currentGame.loc[j, 'PTS']

        # Add data to dataframe
        data = [gameDate, home, homePts, away, awayPts, winLoss]
        dataFrame.loc[i] = data

        i -= 1

    # Reset the index because it currently is out of order
    dataFrame = dataFrame.reset_index()
    return dataFrame

def getUpdatedSchedule ():
    # First game of season id was 0022300061
    gameFinder = leaguegamefinder.LeagueGameFinder()
    games = gameFinder.get_data_frames()[0]

    # Get dataframe for each season
    df2021 = getSeason(games, '2021-10-19')
    df2022 = getSeason(games, '2022-10-18')
    dfCurrent = getSeason(games, '2023-10-24')

    # Get individual stats for historical seasons
    df2021['WL'] = df2021['WL'].astype('category')
    df2021['WL'] = df2021['WL'].cat.codes
    df2022['WL'] = df2022['WL'].astype('category')
    df2022['WL'] = df2022['WL'].cat.codes

    df2021Stats = populateScheduleStats(df2021)
    df2022Stats = populateScheduleStats(df2022)

    # Schedule
    schedule = pd.read_csv("schedule.csv")

    # Update Schedule dates to match nba_api format
    for i in range(len(schedule)):
        # Get date from csv
        old = schedule.loc[i, 'Date']

        # Get month, day, and year
        old_mon = old[4:7]
        day = int(old[8:10])
        str_len = len(old)
        year = old[(str_len - 4):str_len]

        # Convert months from text to num
        if (old_mon == "Jan"):
            new_mon = '01'
        elif (old_mon == "Feb"):
            new_mon = '02'
        elif (old_mon == "Mar"):
            new_mon = '03'
        elif (old_mon == "Apr"):
            new_mon = '04'
        elif (old_mon == "May"):
            new_mon = '05'
        elif (old_mon == "Jun"):
            new_mon = '06'
        elif (old_mon == "Jul"):
            new_mon = '07'
        elif (old_mon == "Aug"):
            new_mon = '08'
        elif (old_mon == "Sep"):
            new_mon = '09'
        elif (old_mon == "Oct"):
            new_mon = '10'
        elif (old_mon == "Nov"):
            new_mon = '11'
        else:
            new_mon = '12'

        # Add 0 infront of days less than 10
        if (day < 10):
            day = "0" + str(day)
        else:
            day = str(day)

        new_date = year + "-" + new_mon + "-" + day

        schedule.loc[i, 'Date'] = new_date



    # Get index of last game that was played
    lastRow = dfCurrent.tail(1).index[0]
    i = schedule.loc[(schedule['Date'] == dfCurrent.loc[lastRow, 'Game Date']) & (schedule['Home/Neutral'] == dfCurrent.loc[lastRow, 'Home'])].index[-1]

    # Make sure we go to next date
    date = schedule.loc[i, 'Date']
    nextDate = schedule.loc[i, 'Date']
    while (date == nextDate):
        i += 1
        nextDate = schedule.loc[i, 'Date']

    # Get rest of schedule
    # Loop through remaining index to create new dataframe with required data
    dataFrame = pd.DataFrame(columns=['Game Date', 'Home', 'Home Pts', 'Away', 'Away Pts', 'WL'])
    date = schedule.loc[i, 'Date']
    getDate = schedule.loc[i, 'Date']
    while (getDate == date):
        print(date)
        getDate = schedule.loc[i, 'Date']
        getAway = schedule.loc[i, 'Visitor/Neutral']
        getHome = schedule.loc[i, 'Home/Neutral']
        data = [getDate, getHome, '0', getAway, '0', '-1']
        dataFrame.loc[i] = data
        i += 1

    # Get stats for current season
    dfCurrent['WL'] = dfCurrent['WL'].astype('category')
    dfCurrent['WL'] = dfCurrent['WL'].cat.codes
    dfCurrentPlayed = populateScheduleStats(dfCurrent)
    dfCurrent = pd.concat([dfCurrent, dataFrame], axis=0, ignore_index=True)
    dfCurrentFull = populateScheduleStats(dfCurrent).to_csv('Upcoming_Games.csv')
    full_schedule = pd.concat([df2021Stats, df2022Stats, dfCurrentPlayed], axis=0, ignore_index=True).to_csv('played.csv')

def getHomeAwayStats (df, date, home, away):
    i = 0
    checkDate = df.loc[0, 'Game Date']
    homeW = 0
    awayW = 0
    homeTotal = 0
    awayTotal = 0
    homeVAwayW = 0
    awayVHomeW = 0
    totalV = 0

    homePoints = 0
    awayPoints = 0

    # Loop through every game before this date
    while (checkDate != date):
        i += 1

        checkDate = df.loc[(i+1), 'Game Date']
        checkAway = df.loc[i, 'Away']
        checkHome = df.loc[i, 'Home']
        checkWin = int(df.loc[i, 'WL'])
        checkAwayPoints = int(df.loc[i, 'Away Pts'])
        checkHomePoints = int(df.loc[i, 'Home Pts'])

        # Tally Wins and Losses for both the home and away team and get points
        homeHome = False
        homeAway = False
        if (home == checkAway):
            if (checkWin == 0):
                homeW += 1
            homeTotal += 1
            homePoints += checkAwayPoints
            homeAway = True
        elif (home == checkHome):
            homeW += checkWin
            homeTotal += 1
            homePoints += checkHomePoints
            homeHome = True

        awayHome = False
        awayAway = False
        if (away == checkAway):
            if (checkWin == 0):
                awayW += 1
            awayTotal += 1
            awayPoints += checkAwayPoints
            awayAway = True
        elif (away == checkHome):
            awayW += checkWin
            awayTotal += 1
            awayPoints += checkHomePoints
            awayHome = True

        # Tally number of games won against each other so far
        if (homeHome and awayAway):
            if (checkWin == 1):
                homeVAwayW += 1
            else:
                awayVHomeW += 1
            totalV += 1
        elif (homeAway and awayHome):
            if (checkWin == 1):
                awayVHomeW += 1
            else:
                homeVAwayW += 1
            totalV += 1

    # Get win percentages and averages
    homeWP = 0
    homeAveP = 0
    awayWP = 0
    awayAveP = 0

    if (homeTotal > 0):
        homeWP = homeW/homeTotal
        homeAveP = homePoints / homeTotal

    if (awayTotal > 0):
        awayWP = awayW/awayTotal
        awayAveP = awayPoints / awayTotal

    if (totalV > 0):
        homeVAwayWP = homeVAwayW/totalV
        awayVHomeWP = awayVHomeW/totalV
    else:
        homeVAwayWP = 1
        awayVHomeWP = 1


    # Get differential stats for home team
    wpDiff = homeWP - awayWP
    homeVAwayWPDiff = homeVAwayWP - awayVHomeWP
    pointsDiff = homeAveP - awayAveP

    return wpDiff, homeVAwayWPDiff, pointsDiff

def populateScheduleStats (df):
    dataFrame = pd.DataFrame(columns=['Date', 'Home', 'Away', 'Win Percent Diff', 'WP vs. A Diff', 'Points Diff', 'Result'])

    for i in range(len(df)):
        #print(i)

        result = df.loc[i, 'WL']
        away = df.loc[i, 'Away']
        home = df.loc[i, 'Home']
        date = df.loc[i, 'Game Date']

        wpDiff, homeVAwayWPDiff, pointsDiff = getHomeAwayStats(df, date, home, away)

        data = [date, home, away, wpDiff, homeVAwayWPDiff, pointsDiff, result]
        dataFrame.loc[i] = data


    return dataFrame

# Display prediction with probability
def display (yPred, teams):
    for i in range(len(yPred)):
        winProb = round(yPred[i],2)
        teams = teams.reset_index(drop=True)
        homeTeam = teams.loc[i, 'Home']
        print(f'The {homeTeam} have a probability of {winProb} of winning')

# Update the schedule if needed
getUpdatedSchedule()

#games_df = pd.read_csv("full_schedule.csv", index_col=0)
playedGames_df = pd.read_csv("played.csv", index_col=0)
upcomingGames_df = pd.read_csv("Upcoming_Games.csv", index_col=0)

#print(games_df)
# Train model
msk = np.random.rand(len(playedGames_df)) < 0.8
train_df = playedGames_df[msk]
# For testing accuracy
#test_df = playedGames_df[~msk]
# For actual predictions
test_df = upcomingGames_df.query('Date == "2023-11-18"')

# Train model for logistic regression
xTrain = train_df.drop(columns=['Date', 'Home', 'Away', 'Result'])
yTrain = train_df[['Result']]
xTest = test_df.drop(columns=['Date', 'Home', 'Away', 'Result'])
yTest = test_df[['Result']]

logReg = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                   intercept_scaling=1, class_weight='balanced', random_state=0,
                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0).fit(xTrain, yTrain.values.ravel())


print(logReg.predict(xTrain))
print(logReg.score(xTrain, yTrain))

print(logReg.predict(xTest))
print(logReg.score(xTest, yTest))

yPred = logReg.predict_proba(xTest)
yPred = yPred[:,1]
display(yPred,test_df)

ex = shap.Explainer(logReg.predict, shap.sample(xTest,1000))

shapValues = ex(xTest)

shap.plots.beeswarm(shapValues)

