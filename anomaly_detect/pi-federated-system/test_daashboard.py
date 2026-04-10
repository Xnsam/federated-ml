import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np
import altair as alt

st.set_page_config(page_title="Edge Monitor", layout="wide")
st.title("📡 Live Anomaly Monitor (Local)")

# Use localhost if running outside Docker
SERVER_URL = "http://localhost:8000"

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()

chart_placeholder = st.empty()
status_placeholder = st.empty()


@st.cache_data
def get_data():
    data = {"Timestamp": [], "Score": [], "Client": []}
    for i in range(120):
        time.sleep(1)
        # data["Timestamp"].append(datetime.now().strftime('%H:%M:%S'))

        data["Timestamp"].append(datetime.now())
        data['Client'].append(np.random.choice(["pi-1", 'pi-2', 'pi-3']))
        data["Score"].append(round(np.random.normal(0, 1), 4))
    data = pd.DataFrame(data)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    return data


THRESHOLD = 2
while True:
    try:
        # response = requests.get(f"{SERVER_URL}/stats", timeout=1)
        # data = response.json()

        # # Update Chart
        # with chart_placeholder.container():
        #     st.line_chart(data, x="Timestamp", y="Score", color="Client")
        data = get_data()

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
        st.error(f"Waiting for server... {e}")

    time.sleep(1)