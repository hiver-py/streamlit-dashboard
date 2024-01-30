import mlflow
import pandas as pd
from config import databricks_host, databricks_token, databricks_experiment_path

from prompts import metrics_prompt, metrics_difference_prompt
import os
from constants import Hyperparameters, Metrics, HYPERPARAMETER_LIST, METRIC_LIST, METRIC_DIFFERENCE_LIST
import matplotlib.pyplot as plt
import streamlit as st
from funcs import typewriter
from call_api import invoke_model 
os.environ["DATABRICKS_HOST"] = databricks_host
os.environ["DATABRICKS_TOKEN"] = databricks_token
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(databricks_experiment_path)


# Use sidebar code from below to ingest data between certain filters
#https://cheat-sheet.streamlit.app

# Fetch runs
logged_runs = mlflow.search_runs(run_view_type=1)  # run_view_type=1 fetches active only runs from Databricks


# Sidebar Calendar
st.sidebar.markdown("Select the date range you want to analyze:")
dates = logged_runs['start_time'].dt.date
first_run_date = dates.min()
last_run_date = dates.max()
unique_dates = sorted(list(set(dates)), reverse=True)

start_date, end_date = st.sidebar.date_input(
    "Select your dates:",
    (first_run_date, last_run_date),
    first_run_date,
    last_run_date,
    format="MM.DD.YYYY",
)

GENAI_button = st.sidebar.button("Enable Generative AI Assisted Interpretation", key="GenAIbutton1")


logged_runs = (logged_runs[
                            (logged_runs['start_time'].dt.date >= start_date) &
                            (logged_runs['start_time'].dt.date <= end_date)
])


# Data Preprocessing
# TODO: Functionalize
model_params = logged_runs.filter(regex='^metrics\.(?!Holdout).*$|^params')
model_params = model_params.rename(columns=lambda x: x.replace('metrics.', ''))
model_params = model_params.rename(columns=lambda x: x.replace('params.', ''))
model_params = model_params.dropna(axis=1, how='all')
unique_counts = model_params.nunique()
constant_cols = unique_counts[unique_counts == 1].index
model_params = model_params.drop(constant_cols, axis=1)

model_params[Metrics.AUC_DIFFERENCE] = model_params[Metrics.TRAIN_AUC] - model_params[Metrics.TEST_AUC]
model_params[Metrics.FSCORE_DIFFERENCE] = model_params[Metrics.TRAIN_PRECISION] - model_params[Metrics.TEST_PRECISION]
model_params[Metrics.PRECISION_DIFFERENCE] = model_params[Metrics.TRAIN_PRECISION] - model_params[Metrics.TEST_PRECISION]
model_params[Metrics.RECALL_DIFFERENCE] = model_params[Metrics.TRAIN_RECALL] - model_params[Metrics.TEST_RECALL]

avg_train_fscore = model_params[[Metrics.TRAIN_FSCORE]].mean(axis=0).round(4)
avg_test_fscore = model_params[[Metrics.TEST_FSCORE]].mean(axis=0).round(4)
avg_train_auc = model_params[[Metrics.TRAIN_AUC]].mean(axis=0).round(4)
avg_test_auc = model_params[[Metrics.TEST_AUC]].mean(axis=0).round(4)
avg_train_prec = model_params[[Metrics.TRAIN_PRECISION]].mean(axis=0).round(4)
avg_test_prec = model_params[[Metrics.TEST_PRECISION]].mean(axis=0).round(4)
avg_train_rec = model_params[[Metrics.TRAIN_RECALL]].mean(axis=0).round(4)
avg_test_rec = model_params[[Metrics.TEST_RECALL]].mean(axis=0).round(4)

# Main Body: Summary Statistics
col1, col2, col3, col4 = st.columns(4)
n_rows = model_params.shape[0]

#TODO: Fix the arrow orientation or look for alternatives.
col1.metric("Average Train F1-Score", f"{avg_train_fscore[0]}", f"Test: {avg_test_fscore[0]}", delta_color="inverse")
col2.metric("Average Train AUC", f"{avg_train_auc[0]}", f"Test: {avg_test_auc[0]}", delta_color="inverse")


col3.metric("Average Train Precision", f"{avg_train_prec[0]}", f"Test: {avg_test_prec[0]}", delta_color="inverse")
col4.metric("Average Train Recall", f"{avg_train_rec[0]}", f"Test: {avg_test_rec[0]}", delta_color="inverse")


# Main Body: Model performance plot
comparative = st.checkbox('Comparative View', key='comparative')
if not comparative:
    mean_performance = model_params[[Metrics.TRAIN_FSCORE, Metrics.TEST_FSCORE,
                                     Metrics.TRAIN_PRECISION, Metrics.TEST_PRECISION,
                                     Metrics.TRAIN_RECALL, Metrics.TEST_RECALL,
                                     Metrics.TRAIN_AUC, Metrics.TEST_AUC]].mean(axis=0).to_frame(name="mean")
    st.bar_chart(mean_performance)
    if GENAI_button:
        api_string_performance = str(mean_performance)
        model_output = invoke_model(code=api_string_performance, system_prompt=metrics_prompt)
        typewriter(text=model_output, speed=8)
    else:
        st.write('')
else:
    difference_bar_data = model_params[[Metrics.AUC_DIFFERENCE,
                                        Metrics.FSCORE_DIFFERENCE,
                                        Metrics.PRECISION_DIFFERENCE,
                                        Metrics.RECALL_DIFFERENCE]].mean(axis=0).to_frame(name="mean")
    st.bar_chart(difference_bar_data)
    if GENAI_button:
        api_string_difference = str(difference_bar_data)
        text = invoke_model(code=api_string_difference, system_prompt=metrics_difference_prompt)
        typewriter(text=text, speed=8)
    else:
        st.write('')


# Main Body: Hyperparameters vs performance plot
model_params['n_estimators'] = model_params['n_estimators'].astype(int)
model_params['learning_rate'] = model_params['learning_rate'].astype(float)
model_params['gamma'] = model_params['gamma'].astype(float)

comparative2 = st.checkbox('Comparative View', key='comparative2')
hyperparameter_option = st.selectbox('Select hyperparameter to explore how it related with model performance:', HYPERPARAMETER_LIST)
if not comparative2:
    metric_option = st.multiselect('Select which metric to visualize:', METRIC_LIST)
    st.line_chart(model_params, x=hyperparameter_option, y=metric_option)
else:
    metric_option = st.multiselect('Select which metric to visualize:', METRIC_DIFFERENCE_LIST)
    st.line_chart(model_params, x=hyperparameter_option, y=metric_option)

if GENAI_button:
    typewriter(text=text, speed=8)
else:
    st.write('')

