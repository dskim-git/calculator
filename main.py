# main.py
import math
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="수학 계산기 & 세계인구 분석",
    page_icon="🧮",
    layout="centered",
)

st.markdown(
    """
    <style>
        :root {
            --bg-1: #0b1020;
            --bg-2: #111a33;
            --bg-3: #1b2445;
            --bg-4: #0f172a;
            --card-bg: rgba(255, 255, 255, 0.90);
            --text-main: #17202c;
            --text-sub: #3f4c5d;
            --accent: #4f46e5;
            --accent-dark: #312e81;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 500px at 15% -10%, rgba(99, 102, 241, 0.28), transparent 60%),
                radial-gradient(900px 450px at 90% 0%, rgba(56, 189, 248, 0.20), transparent 58%),
                linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 35%, var(--bg-3) 70%, var(--bg-4) 100%);
            min-height: 100vh;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.16;
            background-image:
                linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: radial-gradient(circle at 50% 15%, black 20%, transparent 78%);
            -webkit-mask-image: radial-gradient(circle at 50% 15%, black 20%, transparent 78%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: 0 20px 50px rgba(2, 8, 23, 0.35);
        }

        h1, h2, h3,
        p,
        label,
        span,
        div,
        .stMarkdown,
        .stCaption {
            color: #eef3ff;
        }

        h1, h2, h3 {
            letter-spacing: 0.02em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(248, 250, 255, 0.95) 0%, rgba(239, 245, 255, 0.94) 100%);
            border-right: 1px solid rgba(79, 70, 229, 0.18);
        }

        /* ─────────────────────────────────────────────
           사이드바 selectbox 가독성 개선
           ───────────────────────────────────────────── */
        [data-testid="stSidebar"] label {
            color: #17202c !important;
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #17202c !important;
            border: 1px solid rgba(79, 70, 229, 0.35) !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: #17202c !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: #17202c !important;
        }

        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
            color: #17202c !important;
            border: 1px solid rgba(79, 70, 229, 0.25) !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="popover"] ul {
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] li {
            color: #17202c !important;
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] li:hover {
            background-color: #eef2ff !important;
            color: #312e81 !important;
        }

        .hero-card {
            background: var(--card-bg);
            border: 1px solid rgba(79, 70, 229, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 10px 26px rgba(2, 8, 23, 0.14);
            color: var(--text-sub) !important;
        }

        .hero-card * {
            color: var(--text-sub) !important;
        }

        .calc-display {
            background: linear-gradient(145deg, #0f1722 0%, #1c2634 100%);
            color: #eaf3ff;
            border-radius: 14px;
            border: 1px solid rgba(234, 243, 255, 0.18);
            padding: 0.8rem 0.9rem;
            min-height: 5.6rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .calc-expression {
            font-size: 0.95rem;
            color: #a9bbd7;
            text-align: right;
            min-height: 1.3rem;
        }

        .calc-result {
            font-size: 1.65rem;
            font-weight: 700;
            text-align: right;
            letter-spacing: 0.03em;
            word-wrap: break-word;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: 1px solid rgba(79, 70, 229, 0.30);
            background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%);
            color: #1f2937;
            font-weight: 700;
            transition: all 120ms ease;
            min-height: 2.65rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: rgba(79, 70, 229, 0.62);
            box-shadow: 0 8px 14px rgba(79, 70, 229, 0.24);
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] > button:active {
            transform: translateY(0);
        }

        .function-help {
            background: rgba(238, 242, 255, 0.90);
            border: 1px solid rgba(79, 70, 229, 0.22);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 1rem;
            color: #1f2937 !important;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧮 수학 계산기 & 🌍 세계인구 분석")
st.markdown(
    """
    <div class="hero-card">
        일반 계산기 스타일의 빠른 연산과 함께, 모듈러/지수/로그/다항함수 그래프,
        직접 입력한 함수 그래프, 연도별 세계 인구 시각화까지 한 번에 사용할 수 있습니다.
    </div>
    """,
    unsafe_allow_html=True,
)

# 분석에 사용할 연도 컬럼들
YEAR_COLUMNS = ["1970", "1980", "1990", "2000", "2010", "2015", "2020", "2022"]

# ─────────────────────────────────────────────
# 사이드바 메뉴
# ─────────────────────────────────────────────
menu = st.sidebar.selectbox(
    "기능 선택",
    [
        "일반 계산기",
        "다항함수 그래프",
        "함수 그래프",
        "연도별 세계인구 분석",
    ],
)

# ─────────────────────────────────────────────
# 안전한 일반 계산기 평가 함수
# ─────────────────────────────────────────────
ALLOWED_CALC_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Constant,
)


def safe_eval(expression: str):
    """사칙/모듈러/거듭제곱 중심의 계산기 수식을 안전하게 평가합니다."""
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_CALC_NODES):
            raise ValueError("허용되지 않은 수식입니다.")

    return eval(
        compile(tree, "<calculator>", "eval"),
        {"__builtins__": {}},
        {},
    )


# ─────────────────────────────────────────────
# 안전한 함수 그래프 평가 함수
# ─────────────────────────────────────────────
ALLOWED_GRAPH_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)

ALLOWED_GRAPH_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
}

ALLOWED_GRAPH_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}


def safe_graph_eval(expression: str, x_values: np.ndarray):
    """
    함수 그래프용 수식을 안전하게 계산합니다.

    사용 가능 예:
    - x**2
    - sin(x)
    - cos(x)
    - log(x)
    - sqrt(x)
    - exp(x)
    - abs(x)
    - x**3 - 2*x + 1
    """
    expression = expression.replace("^", "**")

    tree = ast.parse(expression, mode="eval")

    allowed_names = {"x"} | set(ALLOWED_GRAPH_FUNCTIONS.keys()) | set(ALLOWED_GRAPH_CONSTANTS.keys())

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_GRAPH_NODES):
            raise ValueError("허용되지 않은 수식 구조가 포함되어 있습니다.")

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"'{node.id}'는 사용할 수 없는 이름입니다.")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("함수 호출 형식이 올바르지 않습니다.")

            if node.func.id not in ALLOWED_GRAPH_FUNCTIONS:
                raise ValueError(f"'{node.func.id}' 함수는 사용할 수 없습니다.")

    safe_namespace = {
        "x": x_values,
        **ALLOWED_GRAPH_FUNCTIONS,
        **ALLOWED_GRAPH_CONSTANTS,
    }

    with np.errstate(all="ignore"):
        y_values = eval(
            compile(tree, "<graph_expression>", "eval"),
            {"__builtins__": {}},
            safe_namespace,
        )

    if np.isscalar(y_values):
        y_values = np.full_like(x_values, float(y_values), dtype=float)

    return np.array(y_values, dtype=float)


def normalize_population_columns(df: pd.DataFrame) -> pd.DataFrame:
    """서로 다른 CSV 헤더 형태를 표준 컬럼명으로 정규화합니다."""
    rename_map = {
        "CCA3": "code",
        "Country/Territory": "Country",
        "World Population Percentage": "World Population Percentage",
    }

    for year in YEAR_COLUMNS:
        rename_map[f"{year} Population"] = year

    df = df.rename(columns=rename_map)
    return df


# ─────────────────────────────────────────────
# 0. 공용 함수: 세계 인구 데이터 불러오기
# ─────────────────────────────────────────────
@st.cache_data
def load_population_data():
    """
    world_population.csv 파일을 불러옵니다.
    예상 컬럼:
      - code (ISO3 국가 코드)
      - Country
      - 1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022
      - World Population Percentage
    """
    df = pd.read_csv("world_population.csv")

    # 컬럼 이름 양쪽 공백 제거
    df.columns = [c.strip() for c in df.columns]
    df = normalize_population_columns(df)

    required_base = ["code", "Country"]
    missing = [col for col in required_base if col not in df.columns]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")

    # 연도 및 비율 컬럼 숫자형으로 변환
    for col in YEAR_COLUMNS + ["World Population Percentage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["code"] = df["code"].astype(str).str.upper().str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    return df


# ─────────────────────────────────────────────
# 1️⃣ 일반 계산기
# ─────────────────────────────────────────────
if menu == "일반 계산기":
    st.subheader("일반 계산기")

    if "calc_expression" not in st.session_state:
        st.session_state.calc_expression = ""

    if "calc_result" not in st.session_state:
        st.session_state.calc_result = "0"

    st.markdown(
        f"""
        <div class="calc-display">
            <div class="calc-expression">{st.session_state.calc_expression or '&nbsp;'}</div>
            <div class="calc-result">{st.session_state.calc_result}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    calc_tab, advanced_tab = st.tabs(["키패드 계산", "고급 연산"])

    with calc_tab:
        keys = [
            ["C", "DEL", "%", "÷"],
            ["7", "8", "9", "×"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "^", "="],
        ]

        clicked = None

        for r, row in enumerate(keys):
            cols = st.columns(4)

            for c, key in enumerate(row):
                if cols[c].button(
                    key,
                    use_container_width=True,
                    key=f"calc_{r}_{c}",
                ):
                    clicked = key

        if clicked:
            if clicked == "C":
                st.session_state.calc_expression = ""
                st.session_state.calc_result = "0"
                st.rerun()

            elif clicked == "DEL":
                st.session_state.calc_expression = st.session_state.calc_expression[:-1]
                st.rerun()

            elif clicked == "=":
                if st.session_state.calc_expression.strip():
                    parsed = (
                        st.session_state.calc_expression
                        .replace("×", "*")
                        .replace("÷", "/")
                        .replace("^", "**")
                    )

                    try:
                        value = safe_eval(parsed)
                        st.session_state.calc_result = str(value)
                        st.session_state.calc_expression = str(value)

                    except ZeroDivisionError:
                        st.session_state.calc_result = "Error: 0으로 나눌 수 없습니다"

                    except Exception:
                        st.session_state.calc_result = "Error: 수식을 확인하세요"

                st.rerun()

            else:
                st.session_state.calc_expression += clicked
                st.rerun()

        st.caption("지원 연산: +, -, ×, ÷, %, ^")

    with advanced_tab:
        left, right = st.columns(2)

        with left:
            st.markdown("#### 모듈러")
            mod_a = int(
                st.number_input(
                    "정수 a",
                    value=10,
                    step=1,
                    key="adv_mod_a",
                )
            )
            mod_n = int(
                st.number_input(
                    "양의 정수 n",
                    value=3,
                    min_value=1,
                    step=1,
                    key="adv_mod_n",
                )
            )

            if st.button(
                "a mod n 계산",
                key="adv_mod_btn",
                use_container_width=True,
            ):
                mod_result = mod_a % mod_n
                st.success(f"결과: {mod_result}")
                st.latex(f"{mod_a} \\bmod {mod_n} = {mod_result}")

        with right:
            st.markdown("#### 지수")
            pow_a = st.number_input(
                "밑 a",
                value=2.0,
                key="adv_pow_a",
            )
            pow_b = st.number_input(
                "지수 b",
                value=3.0,
                key="adv_pow_b",
            )

            if st.button(
                "a^b 계산",
                key="adv_pow_btn",
                use_container_width=True,
            ):
                try:
                    pow_result = pow_a ** pow_b
                    st.success(f"결과: {pow_result}")
                    st.latex(f"{pow_a}^{{{pow_b}}} = {pow_result}")

                except OverflowError:
                    st.error("값이 너무 큽니다. 지수를 줄여보세요.")

        st.markdown("---")
        st.markdown("#### 로그")

        log_col1, log_col2, log_col3 = st.columns([1, 1, 0.8])

        with log_col1:
            log_x = st.number_input(
                "진수 a (>0)",
                value=8.0,
                key="adv_log_x",
            )

        with log_col2:
            log_b = st.number_input(
                "밑 b (>0, b≠1)",
                value=2.0,
                key="adv_log_b",
            )

        with log_col3:
            do_log = st.button(
                "log 계산",
                key="adv_log_btn",
                use_container_width=True,
            )

        if do_log:
            if log_x <= 0:
                st.error("a는 0보다 커야 합니다.")

            elif log_b <= 0 or log_b == 1:
                st.error("b는 0보다 크고 1이 아니어야 합니다.")

            else:
                log_result = math.log(log_x, log_b)
                st.success(f"결과: {log_result}")
                st.latex(f"\\log_{{{log_b}}} {log_x} = {log_result}")


# ─────────────────────────────────────────────
# 2️⃣ 다항함수 그래프
# ─────────────────────────────────────────────
elif menu == "다항함수 그래프":
    st.subheader("📈 다항함수 그래프 스튜디오")
    st.caption("1차, 2차, 3차 다항함수의 그래프를 축, 격자, 실근, 도함수와 함께 확인할 수 있습니다.")

    settings_col, view_col = st.columns([1, 1.4])

    with settings_col:
        degree = st.selectbox(
            "다항식 차수",
            [1, 2, 3],
            key="poly_degree",
        )

        if degree == 1:
            a = st.number_input(
                "a (1차항)",
                value=1.0,
                key="poly1_a",
            )
            b = st.number_input(
                "b (상수항)",
                value=0.0,
                key="poly1_b",
            )

            coeffs = [a, b]
            st.latex(f"f(x) = {a}x + {b}")

        elif degree == 2:
            a = st.number_input(
                "a (2차항)",
                value=1.0,
                key="poly2_a",
            )
            b = st.number_input(
                "b (1차항)",
                value=0.0,
                key="poly2_b",
            )
            c = st.number_input(
                "c (상수항)",
                value=0.0,
                key="poly2_c",
            )

            coeffs = [a, b, c]
            st.latex(f"f(x) = {a}x^2 + {b}x + {c}")

        else:
            a = st.number_input(
                "a (3차항)",
                value=1.0,
                key="poly3_a",
            )
            b = st.number_input(
                "b (2차항)",
                value=0.0,
                key="poly3_b",
            )
            c = st.number_input(
                "c (1차항)",
                value=0.0,
                key="poly3_c",
            )
            d = st.number_input(
                "d (상수항)",
                value=0.0,
                key="poly3_d",
            )

            coeffs = [a, b, c, d]
            st.latex(f"f(x) = {a}x^3 + {b}x^2 + {c}x + {d}")

        x_min, x_max = st.slider(
            "x 범위",
            -50,
            50,
            (-10, 10),
            key="poly_x_range",
        )

        resolution = st.slider(
            "샘플 밀도",
            200,
            3000,
            1000,
            step=100,
            key="poly_res",
        )

        trace_mode = st.selectbox(
            "그래프 스타일",
            ["라인", "라인+마커"],
            key="poly_trace",
        )

        show_area = st.checkbox(
            "x축 기준 면적 채우기",
            value=True,
            key="poly_area",
        )

        show_derivative = st.checkbox(
            "도함수 f'(x) 함께 보기",
            value=False,
            key="poly_deriv",
        )

        show_roots = st.checkbox(
            "실근 표시",
            value=True,
            key="poly_roots",
        )

    with view_col:
        x = np.linspace(x_min, x_max, resolution)
        y = np.polyval(coeffs, x)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers" if trace_mode == "라인+마커" else "lines",
                name="f(x)",
                line=dict(color="#176087", width=3),
                marker=dict(size=3, color="#0e354f"),
                fill="tozeroy" if show_area else None,
                fillcolor="rgba(44, 125, 160, 0.18)",
            )
        )

        if show_derivative:
            d_coeffs = np.polyder(coeffs)
            y_prime = np.polyval(d_coeffs, x)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_prime,
                    mode="lines",
                    name="f'(x)",
                    line=dict(color="#d97b29", width=2, dash="dash"),
                )
            )

        if show_roots:
            roots = np.roots(coeffs)
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
            in_range = [r for r in real_roots if x_min <= r <= x_max]

            if in_range:
                root_y = np.zeros(len(in_range))

                fig.add_trace(
                    go.Scatter(
                        x=in_range,
                        y=root_y,
                        mode="markers+text",
                        text=[f"x={r:.3g}" for r in in_range],
                        textposition="top center",
                        marker=dict(size=11, color="#c43d3d", symbol="diamond"),
                        name="실근",
                    )
                )

        fig.add_hline(
            y=0,
            line_color="rgba(29,37,50,0.45)",
            line_width=1.2,
        )

        fig.add_vline(
            x=0,
            line_color="rgba(29,37,50,0.45)",
            line_width=1.2,
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(29,37,50,0.12)",
            zeroline=False,
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(29,37,50,0.12)",
            zeroline=False,
        )

        fig.update_layout(
            title="Polynomial Graph Studio",
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
            margin=dict(l=10, r=10, t=56, b=10),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        summary_vals = {
            "최소 y": float(np.min(y)),
            "최대 y": float(np.max(y)),
            "평균 y": float(np.mean(y)),
        }

        st.write(summary_vals)


# ─────────────────────────────────────────────
# 3️⃣ 함수 그래프
# ─────────────────────────────────────────────
elif menu == "함수 그래프":
    st.subheader("📊 함수 그래프 그리기")
    st.caption("x에 대한 함수를 직접 입력하면 그래프를 그릴 수 있습니다.")

    st.markdown(
        """
        <div class="function-help">
            <b>입력 예시</b><br>
            x**2, x^2, sin(x), cos(x), tan(x), log(x), sqrt(x), exp(x), abs(x), x**3 - 2*x + 1<br><br>
            <b>사용 가능한 함수</b><br>
            sin, cos, tan, arcsin, arccos, arctan, sqrt, log, ln, log10, exp, abs, floor, ceil<br><br>
            <b>사용 가능한 상수</b><br>
            pi, e
        </div>
        """,
        unsafe_allow_html=True,
    )

    graph_col, option_col = st.columns([1.4, 1])

    with option_col:
        function_expression = st.text_input(
            "함수식 f(x)를 입력하세요",
            value="sin(x)",
            key="function_expression",
        )

        x_min = st.number_input(
            "x 최솟값",
            value=-10.0,
            key="func_x_min",
        )

        x_max = st.number_input(
            "x 최댓값",
            value=10.0,
            key="func_x_max",
        )

        resolution = st.slider(
            "샘플 밀도",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
            key="func_resolution",
        )

        show_points = st.checkbox(
            "점 함께 표시",
            value=False,
            key="func_show_points",
        )

        show_area = st.checkbox(
            "x축 기준 면적 채우기",
            value=False,
            key="func_show_area",
        )

        y_auto_range = st.checkbox(
            "y축 범위 자동 설정",
            value=True,
            key="func_y_auto_range",
        )

        if not y_auto_range:
            y_min = st.number_input(
                "y 최솟값",
                value=-10.0,
                key="func_y_min",
            )

            y_max = st.number_input(
                "y 최댓값",
                value=10.0,
                key="func_y_max",
            )

    with graph_col:
        if x_min >= x_max:
            st.error("x 최솟값은 x 최댓값보다 작아야 합니다.")

        else:
            try:
                x = np.linspace(x_min, x_max, resolution)
                y = safe_graph_eval(function_expression, x)

                valid_mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                if len(x_valid) == 0:
                    st.error("그래프를 그릴 수 있는 유효한 점이 없습니다. 함수식이나 x 범위를 확인해 주세요.")

                else:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=x_valid,
                            y=y_valid,
                            mode="lines+markers" if show_points else "lines",
                            name=f"f(x) = {function_expression}",
                            line=dict(color="#176087", width=3),
                            marker=dict(size=4, color="#0e354f"),
                            fill="tozeroy" if show_area else None,
                            fillcolor="rgba(44, 125, 160, 0.18)",
                        )
                    )

                    fig.add_hline(
                        y=0,
                        line_color="rgba(29,37,50,0.45)",
                        line_width=1.2,
                    )

                    fig.add_vline(
                        x=0,
                        line_color="rgba(29,37,50,0.45)",
                        line_width=1.2,
                    )

                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="rgba(29,37,50,0.12)",
                        zeroline=False,
                    )

                    fig.update_yaxes(
                        showgrid=True,
                        gridcolor="rgba(29,37,50,0.12)",
                        zeroline=False,
                    )

                    if not y_auto_range:
                        if y_min >= y_max:
                            st.warning("y 최솟값은 y 최댓값보다 작아야 합니다. 현재는 자동 범위로 표시합니다.")
                        else:
                            fig.update_yaxes(range=[y_min, y_max])

                    fig.update_layout(
                        title=f"Function Graph: f(x) = {function_expression}",
                        xaxis_title="x",
                        yaxis_title="f(x)",
                        template="plotly_white",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.0,
                        ),
                        margin=dict(l=10, r=10, t=56, b=10),
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    summary_vals = {
                        "유효한 점 개수": int(len(x_valid)),
                        "최소 y": float(np.min(y_valid)),
                        "최대 y": float(np.max(y_valid)),
                        "평균 y": float(np.mean(y_valid)),
                    }

                    st.write(summary_vals)

            except SyntaxError:
                st.error("함수식의 문법을 확인해 주세요. 예: x**2, sin(x), log(x)")

            except ValueError as e:
                st.error(f"함수식을 계산할 수 없습니다: {e}")

            except Exception as e:
                st.error(f"그래프를 그리는 중 오류가 발생했습니다: {e}")


# ─────────────────────────────────────────────
# 4️⃣ 연도별 세계인구 분석
# ─────────────────────────────────────────────
elif menu == "연도별 세계인구 분석":
    st.subheader("🌍 연도별 세계인구 분석")

    # 데이터 불러오기
    try:
        df_pop = load_population_data()

    except FileNotFoundError:
        st.error("world_population.csv 파일을 찾을 수 없습니다. main.py와 같은 폴더에 두고 다시 실행해 주세요.")
        st.stop()

    except ValueError as e:
        st.error(f"데이터 컬럼 형식 오류: {e}")
        st.stop()

    # 1) 연도별 인구 지도
    st.markdown("#### 1) 연도별 세계 인구 지도")

    available_years = [y for y in YEAR_COLUMNS if y in df_pop.columns]

    if not available_years:
        st.error("연도 인구 컬럼을 찾을 수 없습니다. CSV 헤더를 확인해 주세요.")
        st.stop()

    year = st.selectbox(
        "연도를 선택하세요",
        options=available_years,
        format_func=lambda y: f"{y}년",
    )

    if st.button("선택 연도 인구 지도 보기", key="year_map"):
        if year not in df_pop.columns:
            st.error(f"{year} 컬럼을 찾을 수 없습니다. CSV 헤더를 확인해 주세요.")

        else:
            df_year = df_pop[["code", "Country", year]].copy()
            df_year = df_year.rename(columns={year: "Population"})
            df_year["Population"] = pd.to_numeric(df_year["Population"], errors="coerce")
            df_year = df_year.dropna(subset=["Population"])

            fig_year = px.choropleth(
                df_year,
                locations="code",
                locationmode="ISO-3",
                color="Population",
                hover_name="Country",
                color_continuous_scale="Viridis",
                labels={"Population": f"Population {year}"},
            )

            fig_year.update_layout(
                title=f"{year}년 세계 인구 분포",
            )

            st.plotly_chart(fig_year, use_container_width=True)

            top10 = df_year.sort_values("Population", ascending=False).head(10)

            st.markdown("#### 상위 10개국 (인구)")
            st.dataframe(top10, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2) 세계 인구 비율(%) 지도")

    if "World Population Percentage" not in df_pop.columns:
        st.error("'World Population Percentage' 열이 CSV에 존재하지 않습니다.")

    else:
        if st.button("세계 인구 비율 지도 보기", key="share_map"):
            df_share = df_pop[["code", "Country", "World Population Percentage"]].copy()

            df_share["World Population Percentage"] = pd.to_numeric(
                df_share["World Population Percentage"],
                errors="coerce",
            )

            df_share = df_share.dropna(subset=["World Population Percentage"])

            fig_share = px.choropleth(
                df_share,
                locations="code",
                locationmode="ISO-3",
                color="World Population Percentage",
                hover_name="Country",
                color_continuous_scale="Plasma",
                labels={"World Population Percentage": "World Population %"},
            )

            fig_share.update_layout(
                title="세계 인구 비율(%) 분포",
            )

            st.plotly_chart(fig_share, use_container_width=True)

    with st.expander("📄 데이터 일부 미리보기"):
        st.dataframe(df_pop.head())


# ─────────────────────────────────────────────
# 사용 안내 공통
# ─────────────────────────────────────────────
with st.expander("ℹ️ 사용 안내"):
    st.markdown(
        """
        - **일반 계산기**: 키패드 사칙연산 + 고급 연산(모듈러/지수/로그) 통합
        - **다항함수 그래프**: 1~3차 다항식 그래프를 축/격자/실근/도함수 포함으로 시각화
        - **함수 그래프**: 사용자가 직접 입력한 함수식의 그래프를 시각화
          - 예: `x**2`, `x^2`, `sin(x)`, `cos(x)`, `log(x)`, `sqrt(x)`, `exp(x)`
        - **연도별 세계인구 분석**:
          - 선택한 연도의 인구수를 세계지도(choropleth)로 시각화
          - 각 나라의 세계 인구 비율(%)을 색으로 표현
        """
    )


if __name__ == "__main__":
    # streamlit run main.py 로 실행
    pass
