from weather_data.weather import get_weather_data
from holidays.holiday import create_dataframe
from parking.parking import get_cars_parked
from canteen.data_analysis import amount_total
from datetime import date, datetime
import pandas as pd

# Canteen data
canteen = amount_total("../data/kdr_transactions", ".xlsx")
canteen.index = pd.to_datetime(canteen.index).date

# Car data
cars = get_cars_parked("../data/parking_data", ".xlsx")

# Define date interval
earliest_date_cars = cars.iloc[0]
earliest_date_canteen = canteen.iloc[0]

earliest_date_from_datasets = earliest_date_cars.name

if earliest_date_canteen.name < earliest_date_cars.name:
    earliest_date_from_datasets = earliest_date_canteen

date_today = date.today()

# Holiday data
holidays = []
for i in range(
    int(earliest_date_from_datasets.year), int(date_today.year) + 1
):
    holidays.append(create_dataframe(i))
holiday = pd.concat(holidays)
holiday.index = pd.to_datetime(holiday.index).date

# Weather data
weather = get_weather_data(
    str(earliest_date_from_datasets),
    str(date_today.strftime("%Y-%m-%d")),
    "../config.ini",
)
weather.index = pd.to_datetime(weather.index).date
weather.index.name = "date"

# Merge weather and holiday through left outer join
pd.merge(weather, holiday, left_index=True, right_index=True, how="left")
