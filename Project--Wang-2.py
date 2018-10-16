import pandas as pd


def airlineForecast(trainingDataFile, validationDataFile):
    # read trainingDataFile and add days_prior column and day_of_week column
    training = pd.read_csv(trainingDataFile, sep=',', header=0)
    training["days_prior"] = pd.to_datetime(training["departure_date"]) - pd.to_datetime(training["booking_date"])
    training["day_of_week"] = pd.to_datetime(training["departure_date"]).dt.weekday_name

    # get final demand for each departure date and add final_demand column to training data frame
    final_demand = training.loc[training["days_prior"] == "0 days",["departure_date","cum_bookings"]]
    training_set = training.merge(final_demand, left_on="departure_date", right_on="departure_date")
    training_set = training_set.rename(columns = {"cum_bookings_y":"final_demand"})

    # get remaining demand for each days prior and add the column
    training_set["remaining_demand"] = training_set["final_demand"] - training_set["cum_bookings_x"]
    # get booking rate for each days prior and add the column
    training_set["booking_rate"] = training_set["cum_bookings_x"] / training_set["final_demand"]

    # get average remaining demand by days prior and day of week
    forecast_remaining_demand = training_set.groupby(["days_prior", "day_of_week"], as_index=False)["remaining_demand"].mean()
    # get average booking rate by days prior and day of week
    average_booking_rate = training_set.groupby(["days_prior", "day_of_week"], as_index=False)["booking_rate"].mean()

    # read validationDataFile and add days_prior column and day_of_week column
    validation = pd.read_csv(validationDataFile, sep=',', header=0)
    validation["days_prior"] = (pd.to_datetime(validation["departure_date"]) - pd.to_datetime(validation["booking_date"]))
    validation["day_of_week"]= pd.to_datetime(validation["departure_date"]).dt.weekday_name

    # merge average remaining demand series and average booking rate series into the validation data frame
    validation_set = validation.merge(forecast_remaining_demand, left_on=["days_prior", "day_of_week"], right_on=["days_prior", "day_of_week"])
    validation_set = validation_set.rename(columns={"remaining_demand":"forecast_remaining_demand"})
    validation_set = validation_set.merge(average_booking_rate, left_on=["days_prior", "day_of_week"], right_on=["days_prior", "day_of_week"])
    validation_set = validation_set.rename(columns={"booking_rate": "average_booking_rate"})

    # get two forecast final demand results using additive and multiplicative model respectively
    validation_set["forecast_final_demand_addi"] = validation_set["cum_bookings"] + validation_set["forecast_remaining_demand"]
    validation_set["forecast_final_demand_multi"] = validation_set["cum_bookings"] / validation_set["average_booking_rate"]

    # get total errors from naive model, additive model and multiplicative model respectively and calculate MASE of additve and multiplicative models
    total_error_naive = abs(validation_set["naive_forecast"] - validation_set["final_demand"])
    total_error_addi = abs(validation_set["forecast_final_demand_addi"] - validation_set["final_demand"])
    total_error_multi = abs(validation_set["forecast_final_demand_multi"] - validation_set["final_demand"])
    MASE_addi = total_error_addi.sum() / total_error_naive.sum()
    MASE_multi = total_error_multi.sum() / total_error_naive.sum()

    # print result of the model with lower MASE
    output = []
    if MASE_addi < MASE_multi:
        judge = "Additive method has lower MASE :" + str(min([MASE_addi, MASE_multi]))
        result = validation_set[["departure_date", "booking_date", "forecast_final_demand_addi"]]
    elif MASE_addi == MASE_multi:
        judge = "Additive method and Multiplicative method has the same MASE" + str(min([MASE_addi, MASE_multi]))
        result = validation_set[["departure_date", "booking_date", "forecast_final_demand_addi","forecast_final_demand_multi", ]]
    else:
        judge = "Multiplicative method has lower MASE" + str(min([MASE_addi, MASE_multi]))
        result = validation_set[["departure_date", "booking_date", "forecast_final_demand_multi"]]
    output.append(judge)
    output.append(result)
    return output

print airlineForecast("airline_booking_trainingData.csv", "airline_booking_validationData.csv")






