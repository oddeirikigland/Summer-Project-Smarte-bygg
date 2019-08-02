# Smarte Bygg - Prediction models

The main purpose of this project is to predict how many people that will come to work at Telenor Fornebu, for up to approximatly one week into the future. The number of transactions made in the canteens, as well as parking information for the time period October 2016 to February 2019, were provided. It was assumed that the number of people at work equals the number that eats in the canteens. The data set did not contain information about the canteen transactions for all the dates between the start date and end date. Based on the correlation between the parking data and the canteen transactions we were able to deduce how many people were at Telenor when canteen data were missing.

[Final project report](http://htmlpreview.github.io/?https://github.com/telenorbusiness/Summer-Project-Smarte-bygg/blob/master/models/all_models.html)

![GitHub issues](https://img.shields.io/github/issues/telenorbusiness/Summer-Project-Smarte-bygg) ![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/telenorbusiness/Summer-Project-Smarte-bygg) ![GitHub language count](https://img.shields.io/github/languages/count/telenorbusiness/Summer-Project-Smarte-bygg) ![GitHub repo size](https://img.shields.io/github/repo-size/telenorbusiness/Summer-Project-Smarte-bygg) ![GitHub commit activity](https://img.shields.io/github/commit-activity/m/telenorbusiness/Summer-Project-Smarte-bygg) ![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

## Installation

This project assumes you are using Anaconda, with Python version 3.7.3. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

Some may experience problems installing `fbprophet`, if so install with the following command.

```bash
conda install -c conda-forge fbprophet
```

To get weather data from the [Frost API](https://frost.met.no/index.html) you need to register a profile. The id and secret key from that profile needs to be saved in the `config.ini` file in the projects root folder. File format example below.

```ini
[frost_client]
client_id = x
client_secret = x
```

## Usage

All models are stored in [models/saved_models](https://github.com/telenorbusiness/Summer-Project-Smarte-bygg/tree/master/models/saved_models), so it's possible to run our [main jupyter file](https://github.com/telenorbusiness/Summer-Project-Smarte-bygg/blob/master/models/all_models.ipynb) and play around with it. Enjoy!