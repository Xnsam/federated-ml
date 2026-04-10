import streamlit as st
import requests 
import pandas as pd
import time
import altair as alt

st.set_page_config(page_title="Federated Edge Monitor", layout="wide")
st.title("Live Monitor")

SERVER_URL = "http://fed_server:8000"
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 5.0, 1.0)
THRESHOLD = 2

chart_placeholder = st.empty()
status_placeholder = st.empty()

def get_data():
    response = requests.get(f"{SERVER_URL}/stats", timeout=1)
    scores = response.json()
    data = {"Timestamp": [], "Score": [], "Client": []}

    for client_id in scores:
        for val in scores[client_id]:    
            data["Timestamp"].append(val["timestamp"])
            data["Score"].append(val["score"])
            data["Client"].append(client_id)
        
    data = pd.DataFrame(data)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.sort_values(by="Timestamp", ascending=True)
    return data

while True:
    try:
        # response = requests.get(f"{SERVER_URL}/stats", timeout=1)
        # data = response.json()

        # # Update Chart
        # with chart_placeholder.container():
        #     st.line_chart(data, x="Timestamp", y="Score", color="Client")

        # 1. fetch data from the server
        data = get_data()


        # 2. Build chart 

        hover = alt.selection_single(
            fields=["Timestamp"], 
            nearest=True,
            on="mouseover",
            empty="none"
        )
        base = alt.Chart(data).mark_line().encode(
            x=alt.X('Timestamp:T',
                    axis=alt.Axis(format='%H:%M%:%S', labelAngle=~45),
                    title="Time (HH:MM:SS)"),
            y=alt.Y("Score:Q", title="Anomaly Score"),
            color="Client:N",
            tooltip=["Client", "Score", "Timestamp"]
        )
        points = base.transform_filter(hover).mark_circle(size=65)
        tooltips = (
            alt.Chart(data).mark_rule().encode(
            x="Timestamp",
            y="Score",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("Timestamp", title="TS"),
                alt.Tooltip("Score", title="Loss")
            ],
            ).add_selection(hover)
        )


        # main_lines = base.mark_line()

        threshold_line = alt.Chart(
            pd.DataFrame({'y': [THRESHOLD]})
        ).mark_rule(
            color="red",
            strokeDash=[5, 5],
            size=2
        ).encode(y="y:Q")

        chart = (base + threshold_line + points + tooltips).properties(
            height=400
        )
        with chart_placeholder.container():
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Waiting for server data ...{e}")
    
    time.sleep(refresh_rate)