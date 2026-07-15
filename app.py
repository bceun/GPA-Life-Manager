import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# 웹페이지 설정
# =========================
st.set_page_config(
    page_title="Campus Twin",
    page_icon="🎓",
    layout="wide"
)


# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return lr_model, scaler


try:
    lr_model, scaler = load_models()

except Exception as error:
    st.error(
        "모델 파일을 불러오지 못했습니다. "
        "lr_model.pkl과 scaler.pkl 파일을 확인해 주세요."
    )
    st.exception(error)
    st.stop()


# =========================
# 공통 설정
# =========================
FEATURE_COLS = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "screen_time",
    "physical_activity",
    "stress"
]

FEATURE_LABELS = {
    "study_hours": "공부 시간",
    "attendance": "출석률",
    "sleep_hours": "수면 시간",
    "screen_time": "스크린타임",
    "physical_activity": "신체활동",
    "stress": "스트레스"
}

FEATURE_UNITS = {
    "study_hours": "시간",
    "attendance": "%",
    "sleep_hours": "시간",
    "screen_time": "시간",
    "physical_activity": "시간",
    "stress": "단계"
}


# 민감도 분석에서 사용할 변화 방향과 크기
SENSITIVITY_SETTINGS = {
    "study_hours": {
        "change": 1.0,
        "min": 0.0,
        "max": 12.0,
        "description": "공부 시간 1시간 증가"
    },
    "attendance": {
        "change": 5.0,
        "min": 0.0,
        "max": 100.0,
        "description": "출석률 5% 증가"
    },
    "sleep_hours": {
        "change": 1.0,
        "min": 0.0,
        "max": 12.0,
        "description": "수면 시간 1시간 증가"
    },
    "screen_time": {
        "change": -1.0,
        "min": 0.0,
        "max": 16.0,
        "description": "스크린타임 1시간 감소"
    },
    "physical_activity": {
        "change": 1.0,
        "min": 0.0,
        "max": 8.0,
        "description": "신체활동 1시간 증가"
    },
    "stress": {
        "change": -1.0,
        "min": 1.0,
        "max": 10.0,
        "description": "스트레스 1단계 감소"
    }
}


# =========================
# 공통 함수
# =========================
def make_input_df(user_values):
    """
    사용자 입력 딕셔너리를 모델 입력용 데이터프레임으로 변환한다.
    모델 학습 당시 컬럼 순서를 유지한다.
    """
    input_df = pd.DataFrame([user_values])

    missing_cols = [
        col for col in FEATURE_COLS
        if col not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"필요한 입력 컬럼이 없습니다: {missing_cols}"
        )

    return input_df[FEATURE_COLS]


def predict_gpa(input_df, model, scaler):
    """
    입력값을 스케일링한 뒤 GPA를 예측한다.
    예측값은 0.0~4.0 범위로 제한한다.
    """
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)

    if isinstance(pred, np.ndarray):
        pred = pred.reshape(-1)[0]

    pred = np.clip(pred, 0.0, 4.0)

    return float(pred)


def validate_daily_hours(user_values):
    """
    주요 시간형 입력값의 합이 지나치게 큰지 확인한다.
    """
    total_hours = (
        user_values["study_hours"]
        + user_values["sleep_hours"]
        + user_values["screen_time"]
        + user_values["physical_activity"]
    )

    return total_hours


def analyze_sensitivity(user_values, model, scaler):
    """
    현재 입력값에서 변수 하나씩 변경했을 때
    모델의 GPA 예측값이 얼마나 달라지는지 계산한다.
    """
    base_df = make_input_df(user_values)
    base_gpa = predict_gpa(base_df, model, scaler)

    results = []

    for feature, setting in SENSITIVITY_SETTINGS.items():
        changed_values = user_values.copy()

        changed_value = (
            changed_values[feature]
            + setting["change"]
        )

        changed_value = float(
            np.clip(
                changed_value,
                setting["min"],
                setting["max"]
            )
        )

        changed_values[feature] = changed_value

        changed_df = make_input_df(changed_values)
        changed_gpa = predict_gpa(
            changed_df,
            model,
            scaler
        )

        results.append({
            "생활요인 변화": setting["description"],
            "현재 값": user_values[feature],
            "변경 값": changed_value,
            "변경 후 예상 GPA": changed_gpa,
            "예측 변화량": changed_gpa - base_gpa
        })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(
        "예측 변화량",
        ascending=False
    ).reset_index(drop=True)


def compare_plans(
    current_values,
    future_values,
    model,
    scaler
):
    """
    현재 생활계획과 사용자가 설정한 미래 생활계획을 비교한다.
    """
    current_df = make_input_df(current_values)
    future_df = make_input_df(future_values)

    current_gpa = predict_gpa(
        current_df,
        model,
        scaler
    )

    future_gpa = predict_gpa(
        future_df,
        model,
        scaler
    )

    comparison_rows = []

    for feature in FEATURE_COLS:
        comparison_rows.append({
            "생활요인": FEATURE_LABELS[feature],
            "현재": current_values[feature],
            "미래 계획": future_values[feature],
            "변화": (
                future_values[feature]
                - current_values[feature]
            ),
            "단위": FEATURE_UNITS[feature]
        })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "current_gpa": current_gpa,
        "future_gpa": future_gpa,
        "gpa_change": future_gpa - current_gpa,
        "comparison_df": comparison_df
    }


# =========================
# 세션 상태 초기화
# =========================
if "current_values" not in st.session_state:
    st.session_state.current_values = None

if "lr_prediction" not in st.session_state:
    st.session_state.lr_prediction = None

if "dnn_prediction" not in st.session_state:
    st.session_state.dnn_prediction = None

if "sensitivity_df" not in st.session_state:
    st.session_state.sensitivity_df = None


# =========================
# 웹페이지 상단
# =========================
st.title("🎓 Campus Twin")

st.subheader(
    "AI 기반 대학생활 시뮬레이션 및 생활계획 분석"
)

st.write("""
현재 생활습관을 바탕으로 예상 GPA를 확인하고,
생활요인을 변화시켰을 때 모델의 예측값이 어떻게 달라지는지
가상으로 비교할 수 있습니다.

예측 결과는 실제 성적을 보장하거나 인과관계를 의미하지 않으며,
자기관리 참고용으로만 사용해야 합니다.
""")


# =========================
# 탭 구성
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "현재 상태",
    "개인별 분석",
    "미래 시뮬레이션",
    "프로젝트 설명"
])


# =========================
# 1. 현재 상태
# =========================
with tab1:
    st.header("현재 생활습관 입력")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "하루 공부 시간",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            key="current_study"
        )

        attendance = st.number_input(
            "출석률",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            key="current_attendance"
        )

        sleep_hours = st.number_input(
            "하루 수면 시간",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            key="current_sleep"
        )

    with col2:
        screen_time = st.number_input(
            "하루 스크린타임",
            min_value=0.0,
            max_value=16.0,
            value=5.0,
            step=0.5,
            key="current_screen"
        )

        physical_activity = st.number_input(
            "하루 신체활동 시간",
            min_value=0.0,
            max_value=8.0,
            value=1.0,
            step=0.5,
            key="current_activity"
        )

        stress = st.slider(
            "스트레스 수준",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key="current_stress"
        )

    current_values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "screen_time": screen_time,
        "physical_activity": physical_activity,
        "stress": stress
    }

    total_hours = validate_daily_hours(current_values)

    if total_hours > 24:
        st.warning(
            f"공부·수면·스크린타임·신체활동의 합이 "
            f"{total_hours:.1f}시간입니다. "
            "하루 24시간을 초과하므로 입력값을 다시 확인해 주세요."
        )

    if st.button(
        "현재 상태 분석하기",
        type="primary"
    ):
        try:
            input_data = make_input_df(current_values)

            lr_pred = predict_gpa(
                input_data,
                lr_model,
                scaler
            )

            dnn_pred = predict_gpa(
                input_data,
                dnn_model,
                scaler
            )

            sensitivity_df = analyze_sensitivity(
                current_values,
                lr_model,
                scaler
            )

            st.session_state.current_values = (
                current_values.copy()
            )
            st.session_state.lr_prediction = lr_pred
            st.session_state.dnn_prediction = dnn_pred
            st.session_state.sensitivity_df = sensitivity_df

        except Exception as error:
            st.error(
                "예측 중 오류가 발생했습니다."
            )
            st.exception(error)

    if st.session_state.lr_prediction is not None:
        st.divider()
        st.subheader("예측 결과")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "기본 예상 GPA",
                f"{st.session_state.lr_prediction:.2f}"
            )

        with result_col2:
            st.metric(
                "DNN 비교 예측값",
                f"{st.session_state.dnn_prediction:.2f}"
            )

        model_difference = abs(
            st.session_state.lr_prediction
            - st.session_state.dnn_prediction
        )

        st.write(
            f"두 모델의 예측값 차이: "
            f"**{model_difference:.3f}**"
        )

        st.info("""
        현재 서비스에서는 Linear Regression 예측값을
        기본 참고값으로 사용하고,
        DNN 결과는 모델 간 비교 정보로 제공합니다.
        """)

        st.caption(
            "모델의 예측값은 실제 성적을 확정하지 않습니다."
        )


# =========================
# 2. 개인별 분석
# =========================
with tab2:
    st.header("개인별 민감도 분석")

    st.write("""
    현재 입력값에서 생활요인 하나를 조금씩 변경했을 때,
    모델의 예상 GPA가 얼마나 달라지는지 계산합니다.
    """)

    if (
        st.session_state.current_values is None
        or st.session_state.sensitivity_df is None
    ):
        st.info(
            "'현재 상태' 탭에서 먼저 생활습관을 입력하고 "
            "'현재 상태 분석하기' 버튼을 눌러 주세요."
        )

    else:
        sensitivity_df = (
            st.session_state.sensitivity_df.copy()
        )

        display_df = sensitivity_df.copy()

        display_df["현재 값"] = display_df[
            "현재 값"
        ].map(lambda value: f"{value:.1f}")

        display_df["변경 값"] = display_df[
            "변경 값"
        ].map(lambda value: f"{value:.1f}")

        display_df["변경 후 예상 GPA"] = display_df[
            "변경 후 예상 GPA"
        ].map(lambda value: f"{value:.2f}")

        display_df["예측 변화량"] = display_df[
            "예측 변화량"
        ].map(lambda value: f"{value:+.3f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        chart_df = sensitivity_df.set_index(
            "생활요인 변화"
        )

        st.subheader("생활요인별 예측 변화량")

        st.bar_chart(
            chart_df["예측 변화량"]
        )

        best_result = sensitivity_df.iloc[0]

        if best_result["예측 변화량"] > 0:
            st.success(
                f"현재 입력에서는 "
                f"'{best_result['생활요인 변화']}' 시나리오가 "
                f"가장 큰 예측 상승을 보였습니다. "
                f"예상 변화량은 "
                f"{best_result['예측 변화량']:+.3f}입니다."
            )
        else:
            st.warning(
                "현재 설정한 변화 시나리오에서는 "
                "예측 GPA가 상승하는 항목이 확인되지 않았습니다."
            )

        st.caption(
            "민감도 분석은 모델의 예측 반응을 보여주는 것이며, "
            "해당 행동이 실제 GPA 상승의 원인임을 의미하지 않습니다."
        )


# =========================
# 3. 미래 시뮬레이션
# =========================
with tab3:
    st.header("미래 생활계획 시뮬레이션")

    st.write("""
    현재 생활습관과 앞으로 실천할 계획을 비교하여
    모델의 예상 GPA가 어떻게 달라지는지 확인합니다.
    """)

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 생활습관을 분석해 주세요."
        )

    else:
        current = st.session_state.current_values

        st.subheader("미래 계획 입력")

        future_col1, future_col2 = st.columns(2)

        with future_col1:
            future_study = st.number_input(
                "계획 공부 시간",
                min_value=0.0,
                max_value=12.0,
                value=float(current["study_hours"]),
                step=0.5,
                key="future_study"
            )

            future_attendance = st.number_input(
                "계획 출석률",
                min_value=0.0,
                max_value=100.0,
                value=float(current["attendance"]),
                step=1.0,
                key="future_attendance"
            )

            future_sleep = st.number_input(
                "계획 수면 시간",
                min_value=0.0,
                max_value=12.0,
                value=float(current["sleep_hours"]),
                step=0.5,
                key="future_sleep"
            )

        with future_col2:
            future_screen = st.number_input(
                "계획 스크린타임",
                min_value=0.0,
                max_value=16.0,
                value=float(current["screen_time"]),
                step=0.5,
                key="future_screen"
            )

            future_activity = st.number_input(
                "계획 신체활동 시간",
                min_value=0.0,
                max_value=8.0,
                value=float(
                    current["physical_activity"]
                ),
                step=0.5,
                key="future_activity"
            )

            future_stress = st.slider(
                "계획 스트레스 수준",
                min_value=1.0,
                max_value=10.0,
                value=float(current["stress"]),
                step=0.5,
                key="future_stress"
            )

        future_values = {
            "study_hours": future_study,
            "attendance": future_attendance,
            "sleep_hours": future_sleep,
            "screen_time": future_screen,
            "physical_activity": future_activity,
            "stress": future_stress
        }

        future_total_hours = validate_daily_hours(
            future_values
        )

        if future_total_hours > 24:
            st.warning(
                f"미래 계획의 주요 시간 합이 "
                f"{future_total_hours:.1f}시간입니다. "
                "하루 24시간을 초과하므로 계획을 다시 확인해 주세요."
            )

        if st.button(
            "현재 계획과 비교하기",
            type="primary"
        ):
            try:
                comparison_result = compare_plans(
                    current,
                    future_values,
                    lr_model,
                    scaler
                )

                st.divider()
                st.subheader("시뮬레이션 결과")

                metric_col1, metric_col2, metric_col3 = (
                    st.columns(3)
                )

                with metric_col1:
                    st.metric(
                        "현재 예상 GPA",
                        f"{comparison_result['current_gpa']:.2f}"
                    )

                with metric_col2:
                    st.metric(
                        "미래 계획 예상 GPA",
                        f"{comparison_result['future_gpa']:.2f}"
                    )

                with metric_col3:
                    st.metric(
                        "예상 변화",
                        f"{comparison_result['gpa_change']:+.2f}"
                    )

                comparison_df = (
                    comparison_result["comparison_df"].copy()
                )

                comparison_df["현재"] = comparison_df[
                    "현재"
                ].map(lambda value: f"{value:.1f}")

                comparison_df["미래 계획"] = comparison_df[
                    "미래 계획"
                ].map(lambda value: f"{value:.1f}")

                comparison_df["변화"] = comparison_df[
                    "변화"
                ].map(lambda value: f"{value:+.1f}")

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                if comparison_result["gpa_change"] > 0:
                    st.success(
                        "현재 계획보다 미래 계획의 "
                        "예상 GPA가 높게 나타났습니다."
                    )

                elif comparison_result["gpa_change"] < 0:
                    st.warning(
                        "미래 계획의 예상 GPA가 "
                        "현재 계획보다 낮게 나타났습니다."
                    )

                else:
                    st.info(
                        "현재 계획과 미래 계획의 "
                        "예상 GPA가 동일하게 나타났습니다."
                    )

                st.caption(
                    "시뮬레이션 결과는 모델이 학습한 패턴을 "
                    "바탕으로 한 예측이며 실제 결과를 보장하지 않습니다."
                )

            except Exception as error:
                st.error(
                    "미래 계획 비교 중 오류가 발생했습니다."
                )
                st.exception(error)


# =========================
# 4. 프로젝트 설명
# =========================
with tab4:
    st.header("프로젝트 설명")

    st.subheader("프로젝트 목표")

    st.write("""
    Campus Twin은 대학생의 생활습관을 입력받아
    예상 GPA를 계산하고,
    생활계획의 변화가 예측 결과에 미치는 영향을
    가상으로 비교하는 AI 기반 프로토타입입니다.
    """)

    st.subheader("사용 변수")

    feature_info = pd.DataFrame({
        "변수명": [
            "study_hours",
            "attendance",
            "sleep_hours",
            "screen_time",
            "physical_activity",
            "stress"
        ],
        "설명": [
            "하루 공부 시간",
            "출석률",
            "하루 수면 시간",
            "하루 스크린타임",
            "하루 신체활동 시간",
            "스트레스 수준"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("예측 대상")

    st.write("- GPA")

    st.subheader("비교 모델")

    st.write("""
    - Linear Regression
    - DNN Regression
    """)

    st.subheader("평가 지표")

    st.write("""
    - MSE
    - RMSE
    - MAE
    - R²
    """)

    st.subheader("현재 버전의 기능")

    st.write("""
    - 생활습관 기반 GPA 예측
    - Linear Regression과 DNN 예측 비교
    - 생활요인별 민감도 분석
    - 현재 생활과 미래 생활계획 비교
    """)

    st.subheader("한계 및 주의사항")

    st.warning("""
    본 서비스는 학습 데이터의 통계적 패턴을 이용한
    모의 예측 시스템입니다.

    생활습관과 GPA 사이의 인과관계를 증명하지 않으며,
    실제 성적이나 학업 결과를 보장하지 않습니다.
    """)
