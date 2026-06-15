import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 웹페이지 설정
# =========================
st.set_page_config(
    page_title="GPA Life Manager",
    page_icon="🎓",
    layout="centered"
)

# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return lr_model, scaler

lr_model, scaler = load_models()

# =========================
# 공통 함수
# =========================
feature_cols = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "screen_time",
    "physical_activity",
    "stress"
]

def make_input_df(study_hours, attendance, sleep_hours, screen_time, physical_activity, stress):
    return pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "screen_time": [screen_time],
        "physical_activity": [physical_activity],
        "stress": [stress]
    })

def predict_gpa(input_df, model, scaler):
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)

    if isinstance(pred, np.ndarray):
        pred = pred.flatten()[0]

    pred = np.clip(pred, 0, 4)
    return float(pred)

def simulate_improvement(user_values, model, scaler):
    base_df = make_input_df(**user_values)
    base_gpa = predict_gpa(base_df, model, scaler)

    scenarios = []

    scenarios.append({
        "개선 시나리오": "현재 상태",
        "예상 GPA": base_gpa,
        "변화량": 0.0
    })

    changed = user_values.copy()
    changed["study_hours"] = min(changed["study_hours"] + 1, 12)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "공부 시간 1시간 증가",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    changed = user_values.copy()
    changed["attendance"] = min(changed["attendance"] + 5, 100)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "출석률 5% 증가",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    changed = user_values.copy()
    changed["sleep_hours"] = min(changed["sleep_hours"] + 1, 12)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "수면 시간 1시간 증가",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    changed = user_values.copy()
    changed["screen_time"] = max(changed["screen_time"] - 1, 0)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "스크린타임 1시간 감소",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    changed = user_values.copy()
    changed["physical_activity"] = min(changed["physical_activity"] + 1, 8)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "신체활동 1시간 증가",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    changed = user_values.copy()
    changed["stress"] = max(changed["stress"] - 1, 1)
    new_gpa = predict_gpa(make_input_df(**changed), model, scaler)
    scenarios.append({
        "개선 시나리오": "스트레스 1단계 감소",
        "예상 GPA": new_gpa,
        "변화량": new_gpa - base_gpa
    })

    result = pd.DataFrame(scenarios)
    result = result.sort_values("변화량", ascending=False)

    return result

# =========================
# 웹페이지 본문
# =========================
st.title("🎓 GPA Life Manager")
st.subheader("학생 생활습관 기반 GPA 예측 웹사이트")

st.write("""
이 웹사이트는 학생의 생활습관 정보를 입력받아 예상 GPA를 계산하고,
생활 패턴을 점검할 수 있도록 돕는 모의 학습 관리 웹사이트입니다.

예측 결과는 실제 성적을 확정하는 값이 아니라 참고용입니다.
""")

# =========================
# 입력값 받기
# =========================
st.header("1. 생활습관 입력")

study_hours = st.number_input("하루 공부 시간", 0.0, 12.0, 4.0, 0.5)
attendance = st.number_input("출석률", 0.0, 100.0, 85.0, 1.0)
sleep_hours = st.number_input("하루 수면 시간", 0.0, 12.0, 7.0, 0.5)
screen_time = st.number_input("하루 스크린타임", 0.0, 16.0, 5.0, 0.5)
physical_activity = st.number_input("하루 신체활동 시간", 0.0, 8.0, 1.0, 0.5)
stress = st.slider("스트레스 수준", 1.0, 10.0, 5.0, 0.5)

# =========================
# 예측
# =========================
if st.button("예상 GPA 확인하기"):
    input_data = make_input_df(
        study_hours,
        attendance,
        sleep_hours,
        screen_time,
        physical_activity,
        stress
    )

    user_values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "screen_time": screen_time,
        "physical_activity": physical_activity,
        "stress": stress
    }

    lr_pred = predict_gpa(input_data, lr_model, scaler)

    st.header("2. 예측 결과")

    st.metric("예상 GPA", f"{lr_pred:.2f}")

    st.info("""
    최종 실험에서는 Linear Regression과 DNN Regression의 성능 차이가 매우 작았고,
    Linear Regression이 약간 더 낮은 MSE를 보였습니다.
    따라서 이 웹사이트에서는 Linear Regression 예측값을 기본 참고값으로 사용합니다.
    """)

    st.header("3. 생활습관 피드백")

    feedback = []

    if study_hours < 2:
        feedback.append("공부 시간이 낮은 편입니다. 학습 시간을 조금 더 확보해볼 수 있습니다.")

    if attendance < 70:
        feedback.append("출석률이 낮은 편입니다. 수업 참여도를 점검해볼 필요가 있습니다.")

    if sleep_hours < 6:
        feedback.append("수면 시간이 부족한 편입니다. 컨디션 관리를 위해 수면 시간을 점검해보는 것이 좋습니다.")

    if screen_time > 8:
        feedback.append("스크린타임이 높은 편입니다. 학습 집중을 방해할 수 있는 시간 사용 패턴을 점검해볼 수 있습니다.")

    if physical_activity < 0.5:
        feedback.append("신체활동 시간이 낮은 편입니다. 생활 리듬 관리를 위해 가벼운 활동을 추가해볼 수 있습니다.")

    if stress >= 7:
        feedback.append("스트레스 수준이 높은 편입니다. 학습량뿐 아니라 심리적 부담도 함께 관리할 필요가 있습니다.")

    if len(feedback) == 0:
        feedback.append("입력한 생활습관은 전반적으로 안정적인 편입니다. 현재 학습 루틴을 유지하면서 변화를 관찰해볼 수 있습니다.")

    for f in feedback:
        st.write("- " + f)

    st.warning("""
    이 피드백은 인과관계를 의미하지 않습니다.
    모델이 학습한 패턴을 바탕으로 한 자기관리 참고 정보입니다.
    """)

    st.header("4. 생활습관 개선 시뮬레이션")

    simulation_df = simulate_improvement(user_values, lr_model, scaler)

    st.dataframe(simulation_df, use_container_width=True)

    chart_df = simulation_df[simulation_df["개선 시나리오"] != "현재 상태"].copy()
    chart_df = chart_df.set_index("개선 시나리오")

    st.bar_chart(chart_df["변화량"])

st.header("5. 프로젝트 설명")

st.write("""
사용 feature:
- study_hours
- attendance
- sleep_hours
- screen_time
- physical_activity
- stress

Target:
- gpa

비교 모델:
- Linear Regression
- DNN Regression

평가 지표:
- MSE
- RMSE
- MAE
- R²

본 웹사이트는 수업에서 배운 회귀 분석과 DNN Regression을 GPA 예측 문제에 적용한 결과를
간단한 생활습관 관리 서비스 형태로 확장한 것입니다.
""")
