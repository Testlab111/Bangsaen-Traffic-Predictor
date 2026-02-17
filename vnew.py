import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import folium
from streamlit_folium import st_folium

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Bangsaen Traffic Predictor", page_icon="🚗", layout="centered")


# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('decision_tree_model.pkl')
        le_day = joblib.load('le_day.pkl')
        le_traffic = joblib.load('le_traffic.pkl')
        return model, le_day, le_traffic
    except Exception as e:
        st.error(f"❌ ไม่พบไฟล์โมเดลในโฟลเดอร์: {e}")
        return None, None, None


# --- 3. ฟังก์ชันสร้างแผนที่ (อัปเดตพิกัดละเอียด 50 จุด) ---
def create_route_map(prediction_color):
    # พิกัดกึ่งกลางเส้นทางถนนลงหาด
    map_center = [13.2855, 100.9275]
    m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')

    # พิกัด 50 จุดที่ทับแนวถนนจริง
    route_path = [
        [13.286472, 100.939049], [13.286552, 100.938890], [13.286657, 100.938686],
        [13.286774, 100.938473], [13.286864, 100.938270], [13.286956, 100.937981],
        [13.287003, 100.937720], [13.287022, 100.937420], [13.287015, 100.937039],
        [13.287025, 100.935984], [13.286923, 100.931970], [13.286855, 100.930859],
        [13.286853, 100.930823], [13.286820, 100.929140], [13.286637, 100.928063],
        [13.286450, 100.927098], [13.286350, 100.926378], [13.286130, 100.925371],
        [13.286128, 100.925356], [13.286114, 100.925285], [13.286112, 100.925270],
        [13.286071, 100.925042], [13.286057, 100.924928], [13.286022, 100.924667],
        [13.285961, 100.924322], [13.285797, 100.923402], [13.285785, 100.923353],
        [13.285718, 100.922985], [13.285628, 100.922489], [13.285517, 100.921914],
        [13.285373, 100.921148], [13.285260, 100.920531], [13.285108, 100.919726],
        [13.285106, 100.919713], [13.285080, 100.919507], [13.285014, 100.919291],
        [13.284809, 100.918230], [13.284795, 100.918171], [13.284786, 100.918113],
        [13.284782, 100.918094], [13.284747, 100.917923], [13.284744, 100.917910],
        [13.284741, 100.917890], [13.284713, 100.917802], [13.284700, 100.917761],
        [13.284653, 100.917633], [13.284646, 100.917610], [13.284636, 100.917580],
        [13.284416, 100.917050], [13.283849, 100.915914]
    ]

    line_color = '#808080'  # สีเทา Default
    if prediction_color == 'Green':
        line_color = '#28A745'
    elif prediction_color == 'Red':
        line_color = '#FF0000'

    # วาดเส้นทาง
    folium.PolyLine(route_path, color=line_color, weight=12, opacity=0.9).add_to(m)

    # ปักหมุดจุดเริ่มต้นและสิ้นสุด
    folium.Marker(route_path[0], popup="เริ่ม: แยกกาแล็คซี่",
                  icon=folium.Icon(color='blue', icon='car', prefix='fa')).add_to(m)
    folium.Marker(route_path[-1], popup="จบ: วงเวียนบางแสน", icon=folium.Icon(color='red', icon='flag')).add_to(m)

    return m


# --- เริ่มการทำงาน ---
# เช็คสถานะเริ่มต้นใน Session State เพื่อให้ข้อมูลค้างไว้
if 'status_for_map' not in st.session_state:
    st.session_state.status_for_map = "Unknown"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

model, le_day, le_traffic = load_assets()

if model:
    st.title("🚗 คาดการณ์สภาพจราจร บางแสน")
    st.write("เส้นทาง: แยกกาแล็คซี่ ➡️ วงเวียนบางแสน")

    # ส่วนรับข้อมูล
    col1, col2 = st.columns(2)
    with col1:
        known_days = list(le_day.classes_)
        day_input = st.selectbox("เลือกวันเดินทาง", options=known_days)

    with col2:
        available_times = []
        for h in range(10, 19):
            available_times.append(datetime.time(h, 0))
            if h < 18: available_times.append(datetime.time(h, 30))
        time_input = st.selectbox("เลือกเวลาเดินทาง", options=available_times)

    predict_btn = st.button("🚀 ตรวจสอบสภาพจราจร", use_container_width=True)

    # ประมวลผลเมื่อกดปุ่ม
    if predict_btn:
        dep_num = time_input.hour + (time_input.minute / 60.0)
        day_encoded = le_day.transform([day_input])[0]
        X = pd.DataFrame({
            'Day_Encoded': [day_encoded],
            'Departure_Num': [dep_num],
            'min': [20.0], 'max': [40.0], 'avg': [30.0]
        })
        res = model.predict(X)
        prediction = le_traffic.inverse_transform(res)[0]

        # เก็บค่าลง Session State
        st.session_state.status_for_map = prediction
        st.session_state.prediction_result = prediction

    # --- ส่วนการแสดงผล (ดึงค่าจาก Session State มาแสดงเสมอ) ---
    if st.session_state.prediction_result:
        if st.session_state.prediction_result == 'Red':
            st.error(f"### 🚩 ผลการทำนาย: การจราจรหนาแน่น (Red)")
        else:
            st.success(f"### ✅ ผลการทำนาย: การจราจรคล่องตัว (Green)")

    # แสดงแผนที่
    st.divider()
    st.subheader("🗺️ แผนที่พยากรณ์เส้นทางจริง")
    my_map = create_route_map(st.session_state.status_for_map)
    st_folium(my_map, width=700, height=450)

    # แสดง Heatmap (ถ้ามี)
    if os.path.exists('traffic_heatmap_2025.png'):
        st.divider()
        st.subheader("📊 Heatmap สรุปภาพรวมความเสี่ยง")
        st.image('traffic_heatmap_2025.png')