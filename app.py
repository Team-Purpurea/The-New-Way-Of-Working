from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

app = Flask(__name__)

# Load your model and data here
def load_model_and_data():
    # Load the pre-trained model
    model = RandomForestRegressor()

    # Load the modified data
    df_filtered = pd.read_csv('df_latest_waiting_times.csv', low_memory=False)
    df_wouter = pd.read_csv('wouter_routes_updated.csv')

    # Check if the required columns exist
    print("Columns in df_filtered:", df_filtered.columns)
    if 'TrafficSeverityNormalized' not in df_filtered.columns:
        print("Column 'TrafficSeverityNormalized' not found in df_filtered.")
        # Handle the absence of the column
        default_values = {
            'FileSeverity': df_filtered['FileSeverity'].mode()[0],
            'WeatherCode': df_filtered['WeatherCode'].mode()[0]
        }
        features = ['RouteNum', 'FileSeverity', 'WeatherCode', 'DayOfWeek']
    else:
        default_values = {
            'FileSeverity': df_filtered['FileSeverity'].mode()[0],
            'TrafficSeverityNormalized': df_filtered['TrafficSeverityNormalized'].mode()[0],
            'WeatherCode': df_filtered['WeatherCode'].mode()[0]
        }
        features = ['RouteNum', 'FileSeverity', 'TrafficSeverityNormalized', 'WeatherCode', 'DayOfWeek']

    # Example training
    target = 'TimeDifferenceMinutes'
    X_train = df_filtered[features]
    y_train = df_filtered[target]
    model.fit(X_train, y_train)
    
    return model, df_wouter, default_values, features

model, df_wouter, default_values, features = load_model_and_data()

day_name_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
office_day_mapping = {v: k for k, v in day_name_mapping.items()}

def get_base_travel_time(route1, route2):
    matching_rows = df_wouter[(df_wouter['route1'] == route1) & (df_wouter['route2'] == route2)]
    if not matching_rows.empty:
        return matching_rows['base_travel_time'].values[0]
    else:
        return np.nan

def generate_schedule():
    # Initialize schedule list
    schedule = []

    # Predict waiting times for each day of the week for the office
    office_days_of_week = pd.DataFrame({
        'RouteNum': [2] * 5,  # A2 for office
        'FileSeverity': [default_values['FileSeverity']] * 5,
        'TrafficSeverityNormalized': [default_values.get('TrafficSeverityNormalized', np.nan)] * 5,
        'WeatherCode': [default_values['WeatherCode']] * 5,
        'DayOfWeek': range(5)  # Monday to Friday
    })

    office_days_of_week = office_days_of_week[features]  # Ensure correct order
    office_days_of_week['predicted_waiting_time'] = model.predict(office_days_of_week)

    office_days_of_week['base_travel_time'] = get_base_travel_time(2, 0)
    office_days_of_week['total_travel_time'] = office_days_of_week['predicted_waiting_time'] + office_days_of_week['base_travel_time']
    office_days_of_week['Day'] = office_days_of_week['DayOfWeek'].map(office_day_mapping)

    # Find the best day to go to the office
    best_office_day = office_days_of_week.sort_values(by='total_travel_time').iloc[0]

    # Add office day to schedule
    schedule.append({
        "Day": best_office_day['Day'],
        "From": "Home",
        "To": "Office",
        "Route": "A2",
        "Length": df_wouter[(df_wouter['route1'] == 2) & (df_wouter['route2'] == 0)]['total_lenght(km)'].values[0],
        "BaseTravelTime": best_office_day['base_travel_time'],
        "PredictedWaitingTime": best_office_day['predicted_waiting_time'],
        "TotalTravelTime": best_office_day['total_travel_time']
    })

    # Exclude the selected office day from client travel options
    available_days = set(range(5)) - {best_office_day['DayOfWeek']}

    # Predict waiting times for client days
    client_routes = df_wouter[(df_wouter['start'] == 'home') & (df_wouter['arrival'] == 'client')]
    routes = client_routes[['route1', 'route2']].drop_duplicates()
    routes['Route'] = routes.apply(lambda x: f"{x['route1']}+{x['route2']}" if x['route2'] != 0 else str(x['route1']), axis=1)

    client_days_of_week = pd.DataFrame({
        'RouteNum': np.repeat(routes['route1'].values, len(available_days)),
        'FileSeverity': [default_values['FileSeverity']] * len(routes) * len(available_days),
        'TrafficSeverityNormalized': [default_values.get('TrafficSeverityNormalized', np.nan)] * len(routes) * len(available_days),
        'WeatherCode': [default_values['WeatherCode']] * len(routes) * len(available_days),
        'DayOfWeek': np.tile(list(available_days), len(routes)),
        'Route': np.repeat(routes['Route'].values, len(available_days))
    })

    client_days_of_week = client_days_of_week[features + ['Route']]  # Ensure correct order and include 'Route'
    client_days_of_week['predicted_waiting_time'] = model.predict(client_days_of_week[features])

    client_days_of_week['base_travel_time'] = client_days_of_week.apply(
        lambda x: get_base_travel_time(x['RouteNum'], int(x['Route'].split('+')[1]) if '+' in x['Route'] else 0), axis=1
    )
    client_days_of_week = client_days_of_week.dropna(subset=['base_travel_time'])
    client_days_of_week['total_travel_time'] = client_days_of_week['predicted_waiting_time'] + client_days_of_week['base_travel_time']
    client_days_of_week['Day'] = client_days_of_week['DayOfWeek'].map(office_day_mapping)

    # Debugging: Print available days for client travel
    print("Available days for client travel:", available_days)
    print("Client days of week predictions:\n", client_days_of_week[['Day', 'Route', 'total_travel_time']])

    # Find the best two unique days to go to the client
    best_client_days = client_days_of_week.sort_values(by='total_travel_time').drop_duplicates(subset=['DayOfWeek']).head(2)

    for _, best_client_day in best_client_days.iterrows():
        route_length = df_wouter[(df_wouter['route1'] == best_client_day['RouteNum']) & (df_wouter['route2'] == int(best_client_day['Route'].split('+')[1]) if '+' in best_client_day['Route'] else 0)]['total_lenght(km)']
        if not route_length.empty:
            schedule.append({
                "Day": best_client_day['Day'],
                "From": "Home",
                "To": "Client",
                "Route": best_client_day['Route'],
                "Length": route_length.values[0],
                "BaseTravelTime": best_client_day['base_travel_time'],
                "PredictedWaitingTime": best_client_day['predicted_waiting_time'],
                "TotalTravelTime": best_client_day['total_travel_time']
            })

    # Debugging: Print the initial schedule with client days
    print("Initial schedule with client days:\n", schedule)

    # Ensure two unique days for the client
    client_days_added = len([entry for entry in schedule if entry['To'] == 'Client'])
    if client_days_added < 2:
        remaining_client_days = client_days_of_week[~client_days_of_week['Day'].isin([d['Day'] for d in schedule])]
        additional_client_days_needed = 2 - client_days_added
        additional_client_days = remaining_client_days.sort_values(by='total_travel_time').head(additional_client_days_needed)

        # Debugging: Print remaining client days
        print("Remaining client days for selection:\n", remaining_client_days[['Day', 'Route', 'total_travel_time']])

        for _, additional_client_day in additional_client_days.iterrows():
            route_length = df_wouter[(df_wouter['route1'] == additional_client_day['RouteNum']) & (df_wouter['route2'] == int(additional_client_day['Route'].split('+')[1]) if '+' in additional_client_day['Route'] else 0)]['total_lenght(km)']
            if not route_length.empty:
                schedule.append({
                    "Day": additional_client_day['Day'],
                    "From": "Home",
                    "To": "Client",
                    "Route": additional_client_day['Route'],
                    "Length": route_length.values[0],
                    "BaseTravelTime": additional_client_day['base_travel_time'],
                    "PredictedWaitingTime": additional_client_day['predicted_waiting_time'],
                    "TotalTravelTime": additional_client_day['total_travel_time']
                })

    # Debugging: Print the schedule after ensuring two client days
    print("Schedule after ensuring two client days:\n", schedule)

    # Determine home days
    all_days = set(day_name_mapping.keys())
    office_and_client_days = {best_office_day['Day']} | set([d['Day'] for d in schedule if d['To'] == 'Client'])
    home_days = all_days - office_and_client_days

    for day in home_days:
        schedule.append({
            "Day": day,
            "From": "-",
            "To": "-",
            "Route": "-",
            "Length": "-",
            "BaseTravelTime": "-",
            "PredictedWaitingTime": "-",
            "TotalTravelTime": "-"
        })

    # Sort the schedule by day of the week
    schedule = sorted(schedule, key=lambda x: day_name_mapping[x['Day']])

    # Create Plotly plots for office and client
    # Plot for office
    office_fig = go.Figure()
    office_fig.add_trace(go.Scatter(x=office_days_of_week['Day'], y=office_days_of_week['predicted_waiting_time'], mode='lines+markers', name='Predicted Waiting Time', line=dict(color='blue')))
    office_fig.add_trace(go.Scatter(x=office_days_of_week['Day'], y=office_days_of_week['total_travel_time'], mode='lines+markers', name='Total Travel Time', line=dict(color='red', dash='dash')))
    office_fig.add_trace(go.Scatter(x=[best_office_day['Day']], y=[best_office_day['total_travel_time']], mode='markers', name='Best Day to Travel', marker=dict(color='green', size=10)))
    office_fig.update_layout(title='Predicted Waiting Time and Total Travel Time for Each Weekday to Go to Office (A2)', xaxis_title='Day of the Week', yaxis_title='Time (minutes)')

    office_plot_html = office_fig.to_html(full_html=False)

    # Plot for client
    client_fig = go.Figure()
    for route in routes['Route']:
        route_data = client_days_of_week[client_days_of_week['Route'] == route]
        client_fig.add_trace(go.Scatter(x=route_data['Day'], y=route_data['total_travel_time'], mode='lines+markers', name=route))
    client_fig.update_layout(title='Predicted Total Travel Time for Each Weekday and Route to Client', xaxis_title='Day of the Week', yaxis_title='Total Travel Time (minutes)')

    client_plot_html = client_fig.to_html(full_html=False)

    return schedule, office_plot_html, client_plot_html

@app.route('/')
def index():
    schedule, office_plot_html, client_plot_html = generate_schedule()
    print("Final Schedule:", schedule)
    return render_template('schedule.html', schedule=schedule, office_plot_html=office_plot_html, client_plot_html=client_plot_html)

if __name__ == '__main__':
    app.run(debug=True)
