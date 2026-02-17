import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# --- 1. โหลดและเตรียมข้อมูล (เหมือนเดิมเพื่อให้ตัวแปรครบ) ---
file_path = r"C:\Users\modern\OneDrive - BUU\Documents\Data Project\ชุดข้อมูลทั้งหมด 3 ปี.xlsx"
try:
    df = pd.read_excel(file_path)
except:
    df = pd.read_csv(file_path.replace('.xlsx', '.csv'), encoding='cp874')

df = df.dropna(subset=['Day (Saturday or Sunday)', 'Traffic condition (Red or Green)'])
df['Day (Saturday or Sunday)'] = df['Day (Saturday or Sunday)'].astype(str)
df['Traffic condition (Red or Green)'] = df['Traffic condition (Red or Green)'].astype(str)

le_day = LabelEncoder()
le_traffic = LabelEncoder() # <--- ตัวแปรที่เคยหายไป ถูกประกาศตรงนี้

df['Day_Encoded'] = le_day.fit_transform(df['Day (Saturday or Sunday)'])
df['Traffic_Encoded'] = le_traffic.fit_transform(df['Traffic condition (Red or Green)'])

# ฟังก์ชันแปลงเวลา
def time_to_float(time_val):
    try:
        h, m, s = map(int, str(time_val).split(':'))
        return h + m/60.0
    except:
        return 0

df['Departure_Num'] = df['Departure'].apply(time_to_float)

X = df[['Day_Encoded', 'Departure_Num', 'min', 'max', 'avg']]
y = df['Traffic_Encoded']

# --- 2. เทรนโมเดล ---
# แนะนำให้ตั้ง max_depth ไว้สั้นๆ เพื่อให้ภาพต้นไม้ไม่รกจนเกินไป (เช่น 3 หรือ 4)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X, y)

# --- 3. ส่วนวาดภาพ (จุดที่แก้ Error) ---
plt.figure(figsize=(20, 10))  # กำหนดขนาดภาพ

plot_tree(model,
          feature_names=['Day', 'Time', 'min', 'max', 'avg'],
          class_names=list(le_traffic.classes_), # ใช้ le_traffic ที่ประกาศไว้ข้างบน
          filled=True,
          rounded=True,
          fontsize=12)

plt.title("Decision Tree สำหรับทำนายสภาพจราจร (3 ปี)")
plt.savefig('traffic_tree.png', dpi=300) # บันทึกเป็นไฟล์รูปภาพ
plt.show()

print("รันสำเร็จ! ระบบได้บันทึกภาพต้นไม้ไว้ที่ไฟล์ 'traffic_tree.png'")