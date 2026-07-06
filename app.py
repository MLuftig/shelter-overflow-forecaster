 
@st.cache_data
def generate_steady_state_population(n_days=365, seed=42):
    rng = np.random.default_rng(seed)
    current_animals = []
    for day in range(n_days):
        current_animals = [los - 1 for los in current_animals if los - 1 > 0]
        new_arrivals = stats.nbinom.rvs(n_nb, p_nb, random_state=rng)
        for _ in range(new_arrivals):
            if rng.random() < p_long_stay:
                los = stats.gamma.rvs(shape_long, loc=threshold, scale=scale_long, random_state=rng)
            else:
                los = stats.gamma.rvs(shape_short, loc=0, scale=scale_short, random_state=rng)
            current_animals.append(los)
    return current_animals
 
 
def simulate_forecast_period(pressures, capacity, starting_population, rng):
    current_animals = starting_population.copy()
    days_over_capacity = 0
    daily_population = []
 
    for day_pressure in pressures:
        current_animals = [los - 1 for los in current_animals if los - 1 > 0]
        adj_mean = adjusted_intake_mean(day_pressure)
        n_day, p_day = nb_params_from_mean(adj_mean)
        new_arrivals = stats.nbinom.rvs(n_day, p_day, random_state=rng)
 
        for _ in range(new_arrivals):
            if rng.random() < p_long_stay:
                los = stats.gamma.rvs(shape_long, loc=threshold, scale=scale_long, random_state=rng)
            else:
                los = stats.gamma.rvs(shape_short, loc=0, scale=scale_short, random_state=rng)
            current_animals.append(los)
 
        pop_today = len(current_animals)
        daily_population.append(pop_today)
        if pop_today > capacity:
            days_over_capacity += 1
 
    return days_over_capacity, daily_population
 
 
# ============================================================
# Streamlit UI
# ============================================================
st.title("🐾 Shelter Overflow Risk Forecaster")
st.write(
    "Estimate the probability of exceeding shelter capacity this week, "
    "based on a 7-day weather forecast."
)
 
st.subheader("Kennel Capacity")
capacity = st.slider("Shelter capacity (max animals)", 700, 1200, 900, step=10)
 
st.subheader("7-Day Weather Forecast")
st.caption(
    "Pick the expected condition for each day. Lower-pressure conditions "
    "(rain, storms) are associated with higher shelter intake."
)
 
# Plain-language weather conditions mapped to typical sea-level pressure (mb).
# These are representative approximations, not live measurements.
WEATHER_OPTIONS = {
    "☀️ Clear / Sunny": 1023,
    "⛅ Partly Cloudy": 1016,
    "☁️ Overcast": 1010,
    "🌦️ Light Rain": 1005,
    "🌧️ Heavy Rain / Storm": 995,
    "⛈️ Severe Storm": 985,
}
 
cols = st.columns(7)
default_condition = "⛅ Partly Cloudy"
pressures = []
for i, col in enumerate(cols):
    with col:
        condition = st.selectbox(
            f"Day {i + 1}",
            options=list(WEATHER_OPTIONS.keys()),
            index=list(WEATHER_OPTIONS.keys()).index(default_condition),
            key=f"day_{i}",
        )
        pressures.append(WEATHER_OPTIONS[condition])
 
n_simulations = st.slider("Number of simulations", 100, 5000, 1000, step=100)
 
if st.button("Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        starting_population = generate_steady_state_population()
        rng = np.random.default_rng()
 
        results = [
            simulate_forecast_period(pressures, capacity, starting_population, rng)
            for _ in range(n_simulations)
        ]
        days_over = [r[0] for r in results]
        overflow_pct = (np.array(days_over) > 0).mean() * 100
        avg_days_over = np.mean(days_over)
 
    col1, col2 = st.columns(2)
    col1.metric("Probability of overflow this week", f"{overflow_pct:.1f}%")
    col2.metric("Average days over capacity", f"{avg_days_over:.2f}")
 
    all_pops = np.array([r[1] for r in results])
    mean_pop = all_pops.mean(axis=0)
    st.line_chart(mean_pop)
    st.caption("Average simulated shelter population across the forecast week")
 
st.divider()
with st.expander("About this model"):
    st.write(
        """
        This tool combines two pieces of analysis:
 
        1. **Length-of-stay mixture model** — a two-component gamma distribution
           fit to historical shelter data, separating typical short stays from a
           smaller population of long-term residents.
        2. **Weather-adjusted intake model** — a negative binomial regression
           showing that each 1 mb drop in sea-level pressure is associated with
           roughly a 1.3% increase in expected daily intake. Each weather
           condition above (Clear, Rain, Storm, etc.) maps to a typical
           pressure value, so no meteorology knowledge is needed to use the tool.
 
        The simulator runs thousands of randomized 7-day scenarios using these
        fitted distributions to estimate the probability of exceeding a given
        kennel capacity.
 
        **Note:** pressure explains only a small share of day-to-day intake
        variation (many other factors matter too), so this should be read as a
        modest risk adjustment on top of baseline volume, not a precise forecast.
        """
    )
