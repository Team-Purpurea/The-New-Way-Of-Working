from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from flask import jsonify

app = Flask(__name__)

# Define day mappings
day_name_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
inverse_day_name_mapping = {v: k for k, v in day_name_mapping.items()}

# Load your model and data here
def load_model_and_data():
    # Load the pre-trained model
    # model = RandomForestRegressor()
    model = XGBRegressor()

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

day_name_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
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
    office_days_of_week['Day'] = office_days_of_week['DayOfWeek'].map(day_name_mapping)

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
    client_days_of_week['Day'] = client_days_of_week['DayOfWeek'].map(day_name_mapping)

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

    # Ensure two unique days for the client
    client_days_added = len([entry for entry in schedule if entry['To'] == 'Client'])
    if client_days_added < 2:
        remaining_client_days = client_days_of_week[~client_days_of_week['Day'].isin([d['Day'] for d in schedule])]
        additional_client_days_needed = 2 - client_days_added
        additional_client_days = remaining_client_days.sort_values(by='total_travel_time').head(additional_client_days_needed)

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

    # Determine home days
    all_days = set(day_name_mapping.values())
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
    schedule = sorted(schedule, key=lambda x: office_day_mapping[x['Day']])

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

def generate_custom_schedule(preferences):
    print("Received preferences:", preferences)
    # Initialize schedule list
    schedule = []
    days_of_week = list(day_name_mapping.keys())

    # Apply user preferences
    home_days = [inverse_day_name_mapping[day] for day, pref in preferences.items() if pref == 'Home']
    office_day = next((inverse_day_name_mapping[day] for day, pref in preferences.items() if pref == 'Office'), None)
    client_days = [inverse_day_name_mapping[day] for day, pref in preferences.items() if pref == 'Client']

    # Debugging information
    print("Home days:", home_days)
    print("Office day:", office_day)
    print("Client days:", client_days)

    # Check constraints
    if len(home_days) > 2 or (office_day and len(client_days) > 1):
        raise ValueError("Invalid preferences provided")

    # Initialize remaining_days
    remaining_days = list(days_of_week)

    # Logic for creating the schedule based on preferences
    if len(home_days) == 2:
        remaining_days = [day for day in remaining_days if day not in home_days]
        office_day = office_day or min(remaining_days, key=lambda day: get_total_travel_time(day, 2))
        if office_day in remaining_days:
            remaining_days.remove(office_day)
        client_days = client_days or sorted(remaining_days, key=lambda day: get_total_travel_time(day, 1))[:2]
    elif len(home_days) == 1 and not office_day and not client_days:
        remaining_days.remove(home_days[0])
        office_day = min(remaining_days, key=lambda day: get_total_travel_time(day, 2))
        if office_day in remaining_days:
            remaining_days.remove(office_day)
        client_days = sorted(remaining_days, key=lambda day: get_total_travel_time(day, 1))[:2]
    elif len(home_days) == 1 and office_day:
        remaining_days.remove(home_days[0])
        client_days = client_days or sorted([day for day in remaining_days if day != office_day], key=lambda day: get_total_travel_time(day, 1))[:2]
    elif len(home_days) == 1 and client_days:
        remaining_days.remove(home_days[0])
        office_day = office_day or min([day for day in remaining_days if day not in client_days], key=lambda day: get_total_travel_time(day, 2))
        remaining_days = [day for day in remaining_days if day not in client_days and day != office_day]
        client_days = client_days + sorted(remaining_days, key=lambda day: get_total_travel_time(day, 1))[:2-len(client_days)]
    elif office_day and client_days:
        remaining_days = [day for day in remaining_days if day != office_day and day not in client_days]
        client_days = client_days + sorted(remaining_days, key=lambda day: get_total_travel_time(day, 1))[:2-len(client_days)]
    elif office_day:
        remaining_days = [day for day in remaining_days if day != office_day]
        client_days = sorted(remaining_days, key=lambda day: get_total_travel_time(day, 1))[:2]
    elif len(client_days) == 2:
        office_day = min([day for day in remaining_days if day not in client_days], key=lambda day: get_total_travel_time(day, 2))
    elif len(client_days) == 1:
        remaining_days = [day for day in remaining_days if day not in client_days]
        office_day = min(remaining_days, key=lambda day: get_total_travel_time(day, 2))
        if office_day in remaining_days:
            remaining_days.remove(office_day)
        client_days.append(min(remaining_days, key=lambda day: get_total_travel_time(day, 1)))
    else:
        raise ValueError("Invalid preferences provided")

    # Debugging information
    print("Final Office day:", office_day)
    print("Final Client days:", client_days)
    print("Remaining Home days:", [day for day in remaining_days if day not in [office_day] + client_days])

    # Add office day to schedule
    if office_day is not None:
        schedule.append({
            "Day": day_name_mapping[office_day],
            "From": "Home",
            "To": "Office",
            "Route": "A2",
            "Length": df_wouter[(df_wouter['route1'] == 2) & (df_wouter['route2'] == 0)]['total_lenght(km)'].values[0],
            "BaseTravelTime": get_base_travel_time(2, 0),
            "PredictedWaitingTime": get_predicted_travel_time(office_day, 2),
            "TotalTravelTime": get_total_travel_time(office_day, 2)
        })
        if office_day in remaining_days:
            remaining_days.remove(office_day)

    # Add client days to schedule
    client_routes = df_wouter[(df_wouter['start'] == 'home') & (df_wouter['arrival'] == 'client')]
    for client_day in client_days:
        best_route = None
        best_total_travel_time = float('inf')
        for _, route in client_routes.iterrows():
            total_travel_time = get_total_travel_time(client_day, route['route1'], route['route2'])
            if total_travel_time < best_total_travel_time:
                best_total_travel_time = total_travel_time
                best_route = route

        if best_route is not None:
            schedule.append({
                "Day": day_name_mapping[client_day],
                "From": "Home",
                "To": "Client",
                "Route": f"{best_route['route1']}+{best_route['route2']}" if best_route['route2'] != 0 else str(best_route['route1']),
                "Length": best_route['total_lenght(km)'],
                "BaseTravelTime": get_base_travel_time(best_route['route1'], best_route['route2']),
                "PredictedWaitingTime": get_predicted_travel_time(client_day, best_route['route1'], best_route['route2']),
                "TotalTravelTime": best_total_travel_time
            })
        if client_day in remaining_days:
            remaining_days.remove(client_day)

    # Add home days to schedule
    final_home_days = set(home_days)
    if len(client_days) < 2:
        available_days = set(remaining_days) - set(client_days)
        additional_client_days = sorted(available_days, key=lambda day: get_total_travel_time(day, 1))[:2-len(client_days)]
        for day in additional_client_days:
            client_days.append(day)
            best_route = None
            best_total_travel_time = float('inf')
            for _, route in client_routes.iterrows():
                total_travel_time = get_total_travel_time(day, route['route1'], route['route2'])
                if total_travel_time < best_total_travel_time:
                    best_total_travel_time = total_travel_time
                    best_route = route

            if best_route is not None:
                schedule.append({
                    "Day": day_name_mapping[day],
                    "From": "Home",
                    "To": "Client",
                    "Route": f"{best_route['route1']}+{best_route['route2']}" if best_route['route2'] != 0 else str(best_route['route1']),
                    "Length": best_route['total_lenght(km)'],
                    "BaseTravelTime": get_base_travel_time(best_route['route1'], best_route['route2']),
                    "PredictedWaitingTime": get_predicted_travel_time(day, best_route['route1'], best_route['route2']),
                    "TotalTravelTime": best_total_travel_time
                })

    final_home_days.update(set(days_of_week) - {office_day} - set(client_days))
    for home_day in final_home_days:
        schedule.append({
            "Day": day_name_mapping[home_day],
            "From": "-",
            "To": "-",
            "Route": "-",
            "Length": "-",
            "BaseTravelTime": "-",
            "PredictedWaitingTime": "-",
            "TotalTravelTime": "-"
        })

    # Sort the schedule by day of the week
    schedule = sorted(schedule, key=lambda x: inverse_day_name_mapping[x['Day']])

    return schedule


def get_base_travel_time(route1, route2):
    matching_rows = df_wouter[(df_wouter['route1'] == route1) & (df_wouter['route2'] == route2)]
    if not matching_rows.empty:
        return matching_rows['base_travel_time'].values[0]
    else:
        return np.nan

def get_predicted_travel_time(day, route_num, route2=0):
    features = {
        'DayOfWeek': day,
        'RouteNum': route_num,
        'FileSeverity': default_values['FileSeverity'],
        'TrafficSeverityNormalized': default_values['TrafficSeverityNormalized'],
        'WeatherCode': default_values['WeatherCode']
    }
    df = pd.DataFrame([features])
    df = df[['RouteNum', 'FileSeverity', 'TrafficSeverityNormalized', 'WeatherCode', 'DayOfWeek']]  # Ensure correct order of features
    predicted_waiting_time = model.predict(df)[0]
    return predicted_waiting_time

def get_total_travel_time(day, route_num, route2=0):
    base_travel_time = get_base_travel_time(route_num, route2)
    predicted_waiting_time = get_predicted_travel_time(day, route_num, route2)
    total_travel_time = base_travel_time + predicted_waiting_time
    return total_travel_time


# @app.route('/')
# def index():
#     schedule, office_plot_html, client_plot_html = generate_schedule()
#     print("Final Schedule:", schedule)
#     return render_template('schedule.html', schedule=schedule, office_plot_html=office_plot_html, client_plot_html=client_plot_html)

@app.route('/')
def home():
    return render_template('preferences.html', days=day_name_mapping.values())


@app.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    preferences = {day: request.form.get(day) for day in days}
    
    home_count = sum(1 for pref in preferences.values() if pref == 'Home')
    office_count = sum(1 for pref in preferences.values() if pref == 'Office')
    client_count = sum(1 for pref in preferences.values() if pref == 'Client')

    if home_count > 2:
        return jsonify({"error": "You can only select up to 2 home days."}), 400
    if office_count > 1:
        return jsonify({"error": "You can only select 1 office day."}), 400
    if (home_count + office_count + client_count) > 2:
        return jsonify({"error": "You can only select up to 2 preferences in total."}), 400

    try:
        schedule = generate_custom_schedule(preferences)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    schedule_html = render_template('schedule_partial.html', schedule=schedule)
    
    return jsonify(schedule_html)

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    return render_template('preferences.html', days=days)

@app.route('/schedule')
def index():
    schedule, office_plot_html, client_plot_html = generate_schedule()
    print("Final Schedule:", schedule)
    return render_template('schedule.html', schedule=schedule, office_plot_html=office_plot_html, client_plot_html=client_plot_html)

if __name__ == '__main__':
    app.run(debug=True)

    
