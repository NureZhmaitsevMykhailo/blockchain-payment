import requests

test_data = {
  "RevolvingUtilizationOfUnsecuredLines": 0,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 1,
  "DebtRatio": 0.05,
  "MonthlyIncome": 3500,
  "NumberOfOpenCreditLinesAndLoans": 2,
  "NumberOfTimes90DaysLate": 2,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 1,
  "NumberOfDependents": 2
}



response = requests.post("http://127.0.0.1:5000/predict", json=test_data)
print(response.json())
