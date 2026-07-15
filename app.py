import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="Campus Twin",
    page_icon="🎓",
    layout="wide"
)


# =========================
# 모델 및 점수 설정
# =========================

# 원본 데이터의 gpa 최대값
# college_students_habits_1M.csv 기준 약 2.009058
MODEL_TARGET_MIN = 0.0
MODEL_TARGET_MAX = 2.009058

# 사용자에게 표시할 GPA 만점
DISPLAY_GPA_MAX = 4.5


# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_models():
    """
    저장된 선형회귀 모델과 StandardScaler를 불러온다.
    """
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
# 변수 설정
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


# 민감도 분석 시 각 변수에 적용할 변화량
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
    사용자 입력값을 모델 입력용 DataFrame으로 변환한다.
    모델 학습 당시의 컬럼 순서를 유지한다.
    """
    input_df = pd.DataFrame([user_values])

    missing_cols = [
        column
        for column in FEATURE_COLS
        if column not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"필요한 입력 컬럼이 없습니다: {missing_cols}"
        )

    return input_df[FEATURE_COLS]


def convert_to_display_gpa(model_score):
    """
    원본 데이터의 약 0~2.009 점수를
    사용자 표시용 0~4.5 GPA로 선형 환산한다.

    이 값은 실제 학점이 아니라
    데이터 범위를 기준으로 환산한 참고 점수다.
    """
    clipped_score = np.clip(
        model_score,
        MODEL_TARGET_MIN,
        MODEL_TARGET_MAX
    )

    converted_score = (
        (clipped_score - MODEL_TARGET_MIN)
        / (MODEL_TARGET_MAX - MODEL_TARGET_MIN)
        * DISPLAY_GPA_MAX
    )

    return float(
        np.clip(
            converted_score,
            0.0,
            DISPLAY_GPA_MAX
        )
    )


def predict_gpa(input_df):
    """
    입력값을 StandardScaler로 변환한 뒤
    선형회귀 모델의 원본 점수와
    4.5점 환산 점수를 반환한다.
    """
    input_scaled = scaler.transform(input_df)

    prediction = lr_model.predict(input_scaled)

    raw_score = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    raw_score = float(
        np.clip(
            raw_score,
            MODEL_TARGET_MIN,
            MODEL_TARGET_MAX
        )
    )

    display_gpa = convert_to_display_gpa(raw_score)

    return {
        "raw_score": raw_score,
        "display_gpa": display_gpa
    }


def calculate_total_hours(user_values):
    """
    시간형 생활습관 항목의 합을 계산한다.
    """
    return (
        user_values["study_hours"]
        + user_values["sleep_hours"]
        + user_values["screen_time"]
        + user_values["physical_activity"]
    )


def analyze_sensitivity(user_values):
    """
    현재 입력값에서 생활요인을 하나씩 변경하고
    원본 점수와 4.5점 환산 GPA의 변화를 계산한다.
    """
    base_input_df = make_input_df(user_values)
    base_prediction = predict_gpa(base_input_df)

    base_raw_score = base_prediction["raw_score"]
    base_display_gpa = base_prediction["display_gpa"]

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

        changed_input_df = make_input_df(changed_values)
        changed_prediction = predict_gpa(changed_input_df)

        changed_raw_score = changed_prediction["raw_score"]
        changed_display_gpa = changed_prediction["display_gpa"]

        results.append({
            "생활요인 변화": setting["description"],
            "현재 값": user_values[feature],
            "변경 값": changed_value,
            "변경 후 환산 GPA": changed_display_gpa,
            "환산 GPA 변화량": (
                changed_display_gpa
                - base_display_gpa
            ),
            "원본 점수 변화량": (
                changed_raw_score
                - base_raw_score
            )
        })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(
        "환산 GPA 변화량",
        ascending=False
    ).reset_index(drop=True)


def compare_plans(current_values, future_values):
    """
    현재 생활계획과 미래 생활계획의
    모델 예측 결과를 비교한다.
    """
    current_input_df = make_input_df(current_values)
    future_input_df = make_input_df(future_values)

    current_prediction = predict_gpa(current_input_df)
    future_prediction = predict_gpa(future_input_df)

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
        "current_raw_score": current_prediction["raw_score"],
        "future_raw_score": future_prediction["raw_score"],
        "current_display_gpa": current_prediction["display_gpa"],
        "future_display_gpa": future_prediction["display_gpa"],
        "display_gpa_change": (
            future_prediction["display_gpa"]
            - current_prediction["display_gpa"]
        ),
        "comparison_df": comparison_df
    }


# =========================
# 세션 상태 초기화
# =========================
if "current_values" not in st.session_state:
    st.session_state.current_values = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "sensitivity_df" not in st.session_state:
    st.session_state.sensitivity_df = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


# =========================
# 상단 소개
# =========================
st.title("🎓 Campus Twin")

st.subheader(
    "AI 기반 대학생활 시뮬레이션 및 생활계획 분석"
)

st.write("""
현재 생활습관을 바탕으로 학업 성과를 예측하고,
생활계획을 변경했을 때 예측 결과가 어떻게 달라지는지
가상으로 비교할 수 있습니다.

원본 데이터의 GPA 값은 일반적인 4.5점 학점 체계가 아니라
약 0점에서 2.01점 범위로 구성되어 있습니다.
본 서비스에서는 이해를 돕기 위해 이를 4.5점 기준으로 선형 환산하여 표시합니다.
""")

st.warning("""
표시되는 환산 GPA는 실제 학교 성적이나 공식 학점을 의미하지 않습니다.
데이터 범위를 4.5점 기준으로 변환한 참고용 지표입니다.
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
# 탭 1: 현재 상태
# =========================
with tab1:
    st.header("현재 생활습관 입력")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
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

    with input_col2:
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

    total_hours = calculate_total_hours(current_values)

    st.caption(
        f"입력된 주요 생활시간 합계: {total_hours:.1f}시간"
    )

    if total_hours > 24:
        st.warning(
            "공부·수면·스크린타임·신체활동의 합이 "
            "하루 24시간을 초과합니다. "
            "입력값을 다시 확인해 주세요."
        )

    if st.button(
        "현재 상태 분석하기",
        type="primary"
    ):
        try:
            input_df = make_input_df(current_values)
            prediction_result = predict_gpa(input_df)
            sensitivity_df = analyze_sensitivity(
                current_values
            )

            st.session_state.current_values = (
                current_values.copy()
            )

            st.session_state.prediction_result = (
                prediction_result
            )

            st.session_state.sensitivity_df = (
                sensitivity_df
            )

            st.session_state.comparison_result = None

        except Exception as error:
            st.error("예측 중 오류가 발생했습니다.")
            st.exception(error)

    if st.session_state.prediction_result is not None:
        prediction_result = (
            st.session_state.prediction_result
        )

        st.divider()
        st.subheader("예측 결과")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "4.5점 환산 예상 GPA",
                f"{prediction_result['display_gpa']:.2f} / 4.50"
            )

        with result_col2:
            st.metric(
                "모델 원본 예측 점수",
                f"{prediction_result['raw_score']:.3f}"
            )

        st.info("""
        선형회귀 모델에는 학습 당시와 동일하게
        StandardScaler로 변환한 입력값을 전달합니다.

        오른쪽의 원본 점수를 데이터의 전체 범위에 맞추어
        왼쪽의 4.5점 환산 GPA로 변환합니다.
        """)

        st.caption(
            "예측 결과는 실제 학점이나 향후 성적을 "
            "확정하거나 보장하지 않습니다."
        )


# =========================
# 탭 2: 개인별 분석
# =========================
with tab2:
    st.header("개인별 민감도 분석")

    st.write("""
    현재 입력값에서 생활요인을 하나씩 변경했을 때
    4.5점 환산 예상 GPA가 얼마나 달라지는지 계산합니다.
    """)

    if (
        st.session_state.current_values is None
        or st.session_state.sensitivity_df is None
    ):
        st.info(
            "'현재 상태' 탭에서 생활습관을 입력한 뒤 "
            "'현재 상태 분석하기' 버튼을 눌러 주세요."
        )

    else:
        sensitivity_df = (
            st.session_state.sensitivity_df.copy()
        )

        display_df = sensitivity_df.copy()

        display_df["현재 값"] = (
            display_df["현재 값"]
            .map(lambda value: f"{value:.1f}")
        )

        display_df["변경 값"] = (
            display_df["변경 값"]
            .map(lambda value: f"{value:.1f}")
        )

        display_df["변경 후 환산 GPA"] = (
            display_df["변경 후 환산 GPA"]
            .map(lambda value: f"{value:.2f}")
        )

        display_df["환산 GPA 변화량"] = (
            display_df["환산 GPA 변화량"]
            .map(lambda value: f"{value:+.3f}")
        )

        display_df["원본 점수 변화량"] = (
            display_df["원본 점수 변화량"]
            .map(lambda value: f"{value:+.4f}")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("생활요인별 환산 GPA 변화량")

        chart_df = sensitivity_df[
            [
                "생활요인 변화",
                "환산 GPA 변화량"
            ]
        ].copy()

        chart_df = chart_df.set_index(
            "생활요인 변화"
        )

        st.bar_chart(
            chart_df["환산 GPA 변화량"],
            use_container_width=True
        )

        best_result = sensitivity_df.iloc[0]

        if best_result["환산 GPA 변화량"] > 0:
            st.success(
                f"현재 입력에서는 "
                f"'{best_result['생활요인 변화']}' 시나리오가 "
                f"가장 큰 예측 상승을 보였습니다. "
                f"환산 GPA 예상 변화량은 "
                f"{best_result['환산 GPA 변화량']:+.3f}입니다."
            )

        else:
            st.warning(
                "현재 설정된 변화 시나리오에서는 "
                "환산 GPA가 상승하는 항목이 확인되지 않았습니다."
            )

        st.caption(
            "민감도 분석은 모델 예측값의 반응을 보여주는 것이며, "
            "해당 행동이 실제 GPA 변화의 원인임을 의미하지 않습니다."
        )


# =========================
# 탭 3: 미래 시뮬레이션
# =========================
with tab3:
    st.header("미래 생활계획 시뮬레이션")

    st.write("""
    현재 생활습관과 앞으로 실천할 계획을 비교하여
    4.5점 환산 예상 GPA가 어떻게 달라지는지 확인합니다.
    """)

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 "
            "생활습관을 분석해 주세요."
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

        future_total_hours = calculate_total_hours(
            future_values
        )

        st.caption(
            f"미래 계획의 주요 생활시간 합계: "
            f"{future_total_hours:.1f}시간"
        )

        if future_total_hours > 24:
            st.warning(
                "미래 계획의 시간형 항목 합이 "
                "하루 24시간을 초과합니다. "
                "입력값을 다시 확인해 주세요."
            )

        if st.button(
            "현재 계획과 비교하기",
            type="primary"
        ):
            try:
                comparison_result = compare_plans(
                    current,
                    future_values
                )

                st.session_state.comparison_result = (
                    comparison_result
                )

            except Exception as error:
                st.error(
                    "미래 계획 비교 중 오류가 발생했습니다."
                )
                st.exception(error)

        if st.session_state.comparison_result is not None:
            comparison_result = (
                st.session_state.comparison_result
            )

            st.divider()
            st.subheader("시뮬레이션 결과")

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:
                st.metric(
                    "현재 환산 GPA",
                    (
                        f"{comparison_result['current_display_gpa']:.2f}"
                        " / 4.50"
                    )
                )

            with metric_col2:
                st.metric(
                    "미래 계획 환산 GPA",
                    (
                        f"{comparison_result['future_display_gpa']:.2f}"
                        " / 4.50"
                    )
                )

            with metric_col3:
                st.metric(
                    "환산 GPA 예상 변화",
                    (
                        f"{comparison_result['display_gpa_change']:+.2f}"
                    )
                )

            comparison_df = (
                comparison_result["comparison_df"].copy()
            )

            display_comparison_df = (
                comparison_df.copy()
            )

            display_comparison_df["현재"] = (
                display_comparison_df["현재"]
                .map(lambda value: f"{value:.1f}")
            )

            display_comparison_df["미래 계획"] = (
                display_comparison_df["미래 계획"]
                .map(lambda value: f"{value:.1f}")
            )

            display_comparison_df["변화"] = (
                display_comparison_df["변화"]
                .map(lambda value: f"{value:+.1f}")
            )

            st.dataframe(
                display_comparison_df,
                use_container_width=True,
                hide_index=True
            )

            gpa_change = comparison_result[
                "display_gpa_change"
            ]

            if gpa_change > 0:
                st.success(
                    "현재 계획보다 미래 계획의 "
                    "환산 예상 GPA가 높게 나타났습니다."
                )

            elif gpa_change < 0:
                st.warning(
                    "미래 계획의 환산 예상 GPA가 "
                    "현재 계획보다 낮게 나타났습니다."
                )

            else:
                st.info(
                    "현재 계획과 미래 계획의 "
                    "환산 예상 GPA가 동일하게 나타났습니다."
                )

            with st.expander(
                "원본 모델 점수 확인"
            ):
                st.write(
                    "현재 원본 점수:",
                    f"{comparison_result['current_raw_score']:.4f}"
                )

                st.write(
                    "미래 원본 점수:",
                    f"{comparison_result['future_raw_score']:.4f}"
                )

            st.caption(
                "시뮬레이션은 모델이 학습한 통계적 패턴을 "
                "바탕으로 한 참고용 결과입니다."
            )


# =========================
# 탭 4: 프로젝트 설명
# =========================
with tab4:
    st.header("프로젝트 설명")

    st.subheader("프로젝트 목표")

    st.write("""
    Campus Twin은 대학생의 생활습관을 입력받아
    학업 성과를 예측하고,
    미래 생활계획의 변화가 예측 결과에 미치는 영향을
    가상으로 비교하는 웹서비스입니다.
    """)

    st.subheader("사용 변수")

    feature_info = pd.DataFrame({
        "변수명": FEATURE_COLS,
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

    st.subheader("사용 모델")

    st.write("""
    - Linear Regression
    - StandardScaler 기반 입력 표준화
    """)

    st.subheader("점수 표시 방식")

    st.write(f"""
    원본 데이터의 GPA 범위는 약
    {MODEL_TARGET_MIN:.1f}점에서 {MODEL_TARGET_MAX:.3f}점입니다.

    사용자에게 익숙한 형태로 결과를 보여주기 위해
    원본 모델 점수를 {DISPLAY_GPA_MAX:.1f}점 기준으로
    선형 환산하여 표시합니다.
    """)

    st.code(
        """
환산 GPA
= 원본 예측 점수
÷ 원본 데이터 최대 GPA
× 4.5
        """.strip(),
        language="text"
    )

    st.subheader("현재 기능")

    st.write("""
    - 생활습관 기반 학업 성과 예측
    - 4.5점 기준 환산 GPA 제공
    - 개인별 민감도 분석
    - 생활요인별 예측 변화량 시각화
    - 현재 생활과 미래 생활계획 비교
    - 비현실적인 시간 입력 경고
    """)

    st.subheader("한계 및 주의사항")

    st.warning("""
    4.5점 환산 GPA는 실제 대학의 학점 산정 방식과
    동일한 값이 아닙니다.

    본 서비스는 학습 데이터의 통계적 패턴을 이용한
    모의 분석 시스템이며,
    생활습관과 GPA 사이의 인과관계를 증명하지 않습니다.
    """)
