import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta
import hashlib
import hmac

st.set_page_config(page_title="Cloud AI Business Digital Twin", page_icon="☁️", layout="wide")

# Demo-only authentication. For production, use a managed identity provider and hashed secrets.
USERS = {"admin": hashlib.sha256("ChangeMe123!".encode()).hexdigest(),
         "manager": hashlib.sha256("Manager123!".encode()).hexdigest()}

def login():
    st.title("☁️ Cloud AI Business Digital Twin")
    st.caption("AI-powered business risk prediction, what-if simulation, and cloud resource planning")
    with st.form("login_form"):
        st.subheader("Demo sign-in")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        stored = USERS.get(username)
        supplied = hashlib.sha256(password.encode()).hexdigest()
        if stored and hmac.compare_digest(stored, supplied):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid demo credentials.")
    st.info("Demo accounts: admin / ChangeMe123!  |  manager / Manager123!")
    st.warning("Demo authentication only. Do not use these credentials or this setup for production.")

if not st.session_state.get("authenticated"):
    login()
    st.stop()

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Open module", [
        "Business Overview", "Demand Forecast", "What-if Digital Twin",
        "Cloud Architecture Simulator", "Security & SLA", "About & Limitations"
    ])
    st.write(f"Signed in as **{st.session_state.get('username','user')}**")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    n = 730
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    dow = dates.dayofweek
    seasonal = 1 + 0.18 * np.sin(np.arange(n) * 2 * np.pi / 365.25)
    promo = rng.binomial(1, 0.12, n)
    price = np.maximum(8, 20 + rng.normal(0, 1.4, n) - promo * 1.8)
    trend = np.linspace(0, 5, n)
    demand = np.maximum(0, 65 * seasonal + 7 * (dow >= 5) + 18 * promo - 1.5 * (price - 20) + trend + rng.normal(0, 7, n))
    return pd.DataFrame({"date": dates, "units_sold": np.round(demand).astype(int),
                         "price": price.round(2), "promotion": promo})

def load_data():
    st.subheader("Data source")
    uploaded = st.file_uploader("Optional: upload daily sales CSV (date, units_sold; optional price, promotion)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = {"date", "units_sold"}
            if not required.issubset(df.columns):
                st.error("CSV must contain date and units_sold columns.")
                return make_demo_data()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
            df = df.dropna(subset=["date", "units_sold"]).sort_values("date")
            if "price" not in df: df["price"] = 20.0
            if "promotion" not in df: df["promotion"] = 0
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(20)
            df["promotion"] = pd.to_numeric(df["promotion"], errors="coerce").fillna(0)
            if len(df) < 30:
                st.warning("At least 30 valid daily rows are recommended. Using demo data for a reliable demonstration.")
                return make_demo_data()
            return df
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    return make_demo_data()

if "sales_data" not in st.session_state:
    st.session_state.sales_data = make_demo_data()
df = st.session_state.sales_data

def train_and_forecast(data, horizon=30):
    work = data.copy().sort_values("date")
    work["day_index"] = (work["date"] - work["date"].min()).dt.days
    work["dow"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["dayofyear"] = work["date"].dt.dayofyear
    features = ["day_index", "dow", "month", "dayofyear", "price", "promotion"]
    X = work[features].astype(float)
    y = work["units_sold"].astype(float)
    split = max(int(len(work) * 0.8), 1)
    model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2)
    model.fit(X.iloc[:split], y.iloc[:split])
    eval_pred = model.predict(X.iloc[split:]) if split < len(work) else model.predict(X.iloc[-1:])
    eval_y = y.iloc[split:] if split < len(work) else y.iloc[-1:]
    mae = float(mean_absolute_error(eval_y, eval_pred))
    last_date = work["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({
        "date": future_dates,
        "day_index": (future_dates - work["date"].min()).days,
        "dow": future_dates.dayofweek,
        "month": future_dates.month,
        "dayofyear": future_dates.dayofyear,
        "price": float(work["price"].tail(30).median()),
        "promotion": 0
    })
    future["predicted_demand"] = np.maximum(0, model.predict(future[features].astype(float))).round(1)
    return work, future, mae, model

if page == "Business Overview":
    st.title("Business Overview")
    st.write("A decision-support prototype for retail managers. It combines demand forecasting with business what-if analysis and a cloud architecture simulator.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales records", f"{len(df):,}")
    c2.metric("Average daily units", f"{df.units_sold.mean():.0f}")
    c3.metric("Average unit price", f"${df.price.mean():.2f}")
    c4.metric("Total units", f"{df.units_sold.sum():,.0f}")
    st.plotly_chart(px.line(df.tail(180), x="date", y="units_sold", title="Recent daily sales"), use_container_width=True)
    st.markdown("### Main workflow")
    st.markdown("1. Upload historical sales data or use the built-in synthetic dataset.  \n2. Forecast upcoming demand.  \n3. Test inventory, price, supplier delay, and demand scenarios.  \n4. Simulate cloud capacity, load balancing, SLA, and estimated cost.")
    st.caption("All built-in data is synthetic. Results are demonstrations, not business guarantees.")

elif page == "Demand Forecast":
    st.title("AI Demand Forecast")
    st.write("Random Forest regression using calendar features, price, promotion, and historical demand.")
    df = load_data()
    st.session_state.sales_data = df
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    work, future, mae, model = train_and_forecast(df, horizon)
    a,b,c = st.columns(3)
    a.metric("Forecast total units", f"{future.predicted_demand.sum():,.0f}")
    b.metric("Average daily forecast", f"{future.predicted_demand.mean():.1f}")
    c.metric("Holdout MAE (units/day)", f"{mae:.1f}")
    chart = pd.concat([df[["date","units_sold"]].tail(90).rename(columns={"units_sold":"units"}),
                       future[["date","predicted_demand"]].rename(columns={"predicted_demand":"units"})])
    chart["period"] = np.where(chart.date <= df.date.max(), "Historical", "Forecast")
    st.plotly_chart(px.line(chart, x="date", y="units", color="period", title="Historical demand and forecast"), use_container_width=True)
    st.dataframe(future[["date","predicted_demand"]].rename(columns={"predicted_demand":"predicted_units"}), use_container_width=True)
    st.download_button("Download forecast CSV", future.to_csv(index=False).encode(), "demand_forecast.csv", "text/csv")
    st.caption("MAE is calculated on a chronological holdout slice. This educational model needs further validation before real operational use.")

elif page == "What-if Digital Twin":
    st.title("What-if Business Digital Twin")
    st.write("Change assumptions and compare a simple business scenario. Estimates depend on the assumptions you enter.")
    df = st.session_state.sales_data
    work, future, mae, model = train_and_forecast(df, 30)
    base_demand = float(future.predicted_demand.mean())
    c1,c2,c3 = st.columns(3)
    with c1:
        stock = st.number_input("Available stock (units)", min_value=0, value=1100, step=50)
        supplier_delay = st.slider("Supplier delay (days)", 0, 30, 5)
    with c2:
        unit_price = st.number_input("Selling price per unit ($)", min_value=0.1, value=20.0, step=1.0)
        unit_cost = st.number_input("Cost per unit ($)", min_value=0.0, value=11.0, step=0.5)
    with c3:
        demand_change = st.slider("Demand change (%)", -50, 100, 20)
        holding_cost = st.number_input("Holding cost per unsold unit ($)", min_value=0.0, value=0.8, step=0.1)
    adjusted_demand = max(0, base_demand * (1 + demand_change / 100) * 30)
    effective_stock = max(0, stock - supplier_delay * base_demand)
    sold = min(adjusted_demand, effective_stock)
    lost = max(0, adjusted_demand - effective_stock)
    unsold = max(0, effective_stock - sold)
    revenue = sold * unit_price
    product_cost = sold * unit_cost
    inventory_cost = unsold * holding_cost
    estimated_profit = revenue - product_cost - inventory_cost
    a,b,c,d = st.columns(4)
    a.metric("Adjusted 30-day demand", f"{adjusted_demand:,.0f} units")
    b.metric("Potential lost sales", f"{lost:,.0f} units")
    c.metric("Estimated revenue", f"${revenue:,.0f}")
    d.metric("Estimated profit", f"${estimated_profit:,.0f}")
    comparison = pd.DataFrame({"Metric":["Demand","Stock after delay","Units sold","Lost sales","Estimated profit"],
                               "Baseline":[base_demand*30, stock, min(base_demand*30,stock), max(0,base_demand*30-stock), min(base_demand*30,stock)*(unit_price-unit_cost)],
                               "Scenario":[adjusted_demand,effective_stock,sold,lost,estimated_profit]})
    st.plotly_chart(px.bar(comparison, x="Metric", y=["Baseline","Scenario"], barmode="group", title="Baseline vs scenario"), use_container_width=True)
    st.info("This is a simplified scenario model, not a full financial accounting system. Supplier delay reduces usable stock using average forecast demand.")

elif page == "Cloud Architecture Simulator":
    st.title("Cloud Architecture & Resource Management Simulator")
    st.write("Simulate workload distribution and elastic resource allocation. This demonstrates concepts; it does not create real virtual machines.")
    c1,c2,c3 = st.columns(3)
    with c1:
        requests = st.slider("Incoming requests per minute", 100, 10000, 2400, 100)
    with c2:
        server_count = st.slider("Active virtual server instances", 1, 10, 3)
    with c3:
        capacity = st.slider("Capacity per instance (requests/min)", 200, 2000, 1000, 100)
    total_capacity = server_count * capacity
    utilization = requests / total_capacity * 100
    required = int(np.ceil(requests / capacity))
    overloaded = requests > total_capacity
    cost_per_instance_hour = st.number_input("Illustrative cost per instance/hour ($)", min_value=0.0, value=0.08, step=0.01)
    hours = st.number_input("Hours to estimate", min_value=1, max_value=744, value=24)
    instances_needed = min(10, max(1, required))
    a,b,c,d = st.columns(4)
    a.metric("Total capacity", f"{total_capacity:,}/min")
    b.metric("Utilization", f"{utilization:.1f}%")
    c.metric("Instances suggested", str(instances_needed))
    d.metric("Estimated cost", f"${instances_needed*cost_per_instance_hour*hours:.2f}")
    if overloaded:
        st.error("Over capacity: simulated requests exceed current total capacity.")
    elif utilization > 80:
        st.warning("High utilization: consider scaling out.")
    else:
        st.success("Capacity is sufficient under the current assumptions.")
    distribution = [requests // server_count + (1 if i < requests % server_count else 0) for i in range(server_count)]
    server_df = pd.DataFrame({"Virtual instance":[f"VM-{i+1}" for i in range(server_count)],
                              "Requests assigned":distribution,
                              "Capacity":[capacity]*server_count})
    server_df["Utilization (%)"] = server_df["Requests assigned"] / server_df["Capacity"] * 100
    st.plotly_chart(px.bar(server_df, x="Virtual instance", y=["Requests assigned","Capacity"], barmode="group", title="Load balancing across virtual instances"), use_container_width=True)
    st.dataframe(server_df, use_container_width=True)
    st.caption("Capacity, cost, and scaling are illustrative. No cloud resources are provisioned and no real autoscaling occurs.")

elif page == "Security & SLA":
    st.title("Security, SLA & Service Management")
    st.subheader("Demo identity and access")
    st.write("The app demonstrates a login gate. Production systems should use a managed identity provider, strong password hashing, secrets management, HTTPS, and least-privilege roles.")
    st.subheader("Example Service Level Agreement (SLA)")
    uptime = st.slider("Observed uptime for a sample period (%)", 90.0, 100.0, 99.5, 0.1)
    response_ms = st.slider("Average response time (ms)", 50, 3000, 350, 50)
    target_uptime = st.selectbox("Example uptime target", [99.0, 99.5, 99.9, 99.95], index=1)
    target_response = st.number_input("Example response-time target (ms)", min_value=50, value=500, step=50)
    x,y = st.columns(2)
    x.metric("Uptime target", f"{target_uptime:.2f}%", delta=f"{uptime-target_uptime:+.2f} percentage points")
    y.metric("Response target", f"{target_response} ms", delta=f"{target_response-response_ms:+.0f} ms")
    if uptime >= target_uptime and response_ms <= target_response:
        st.success("The sample metrics meet the entered SLA targets.")
    else:
        st.warning("The sample metrics miss at least one entered SLA target.")
    st.caption("All SLA values are user-entered examples, not measured cloud service guarantees.")

else:
    st.title("About & Limitations")
    st.markdown("""
### Project objective
Demonstrate how AI demand forecasting and cloud resource-management concepts can support retail business decisions.

### Tools
- Python, Streamlit, Pandas, NumPy
- Scikit-learn Random Forest regression
- Plotly charts
- Optional cloud deployment through Streamlit Community Cloud or another provider

### Important limitations
- Included sales data is synthetic unless a CSV is uploaded.
- Forecast quality depends on the dataset and needs more robust validation for real use.
- The digital twin is a simplified business scenario simulator.
- Cloud elasticity, load balancing, cost and virtualization are simulated; this project does not provision real cloud infrastructure.
- Demo login is educational only and must be replaced for production use.
""")
    st.subheader("ICC syllabus mapping")
    mapping = pd.DataFrame([
        ["1. Parallel/distributed computing","Architecture overview and cloud/distributed-computing explanation"],
        ["2. Virtualization","Virtual server instances represented in the simulator; hypervisors explained in report"],
        ["3. Cloud basics","Cloud-hosted app, SaaS/PaaS/IaaS discussion, cloud benefits and risks"],
        ["4. Fundamental architecture","Resource pooling, elasticity, scalability and load balancing simulation"],
        ["5. Advanced architectures","Load-balanced instances and service capacity simulation; advanced relocation/zero downtime discussed"],
        ["6. Cloud security","Demo login, security limitations, identity/access-control discussion"],
        ["7. Service management","Example SLA and cost/billing estimate; AWS/Azure/GCP/OpenStack case study in report"]
    ], columns=["Unit","Project coverage"])
    st.dataframe(mapping, use_container_width=True)
    st.info("For full syllabus coverage, submit a report with diagrams and case studies for topics not implemented directly.")
