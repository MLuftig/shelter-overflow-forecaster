Shelter Overflow Risk Forecaster

A Monte Carlo simulation tool that estimates the probability of a shelter exceeding
kennel capacity in the coming week, adjusted for forecasted weather conditions.

Live app: shelter-overflow-forecaster.streamlit.app

Problem

Shelter intake volume is inherently random, and prior work in this portfolio
(moon-phase & weather analysis)
identified that barometric pressure drops are a statistically significant driver
of intake surges. But knowing an effect is significant doesn't tell you how much
it actually matters operationally — shelter directors need an actual risk estimate,
not a p-value.

Solution

This project quantifies the weather effect and builds it into a stochastic
capacity-planning model:

Length-of-stay mixture model — historical intake data showed a clearly
bimodal length-of-stay distribution: most animals turn over quickly, but a
small (~3%) subpopulation stays for months. Modeling this as a single
distribution understated both groups, so two gamma distributions were fit
separately and combined into a weighted mixture.
Weather-adjusted intake model — a Negative Binomial regression (with an
explicitly estimated dispersion parameter, since the default assumption
understated real overdispersion in the data) found that each 1 mb drop in
sea-level pressure is associated with a ~1.3% increase in expected daily
intake (p < 0.001).
Monte Carlo simulation — thousands of randomized 7-day scenarios are run
using these fitted distributions, tracking daily shelter population against
a user-defined capacity to estimate overflow probability.
Analysis Notebooks

The parameters used in app.py were derived from real Austin Animal Center
shelter intake/outcome data, not assumed or hardcoded arbitrarily. The full
derivation is reproducible from these notebooks:

01-load-and-summarize-inputs.ipynb —
fits the two-component length-of-stay gamma mixture and the baseline daily
arrival Negative Binomial distribution from real shelter data.
02-weather-intake-regression.ipynb —
fits the Negative Binomial regression quantifying the barometric pressure
effect on daily intake, including diagnosing and correcting an initial
convergence failure caused by a scale mismatch between pressure and intake
count magnitudes.
03-build-simulation.ipynb — validates the
fitted parameters against Little's Law, builds the forecast-period simulation
function, and confirms the headline impact numbers below against real
Monte Carlo output.
Impact

At a capacity close to the shelter's typical steady-state population, a
forecasted week of low-pressure storm systems nearly doubled the
probability of at least one overflow day (34% → 61% in testing) — a
meaningful, quantified operational risk that a shelter director could act on
before it happens, not after.

The analysis also showed this effect is not constant: it's most pronounced
when capacity is already tight, and largely disappears when there's enough
buffer above average population. That nuance — weather risk depends on how
much slack you already have — is arguably more useful than a blanket
"weather predicts overflow" claim.

How It Works
Set a kennel capacity
Select a weather condition (Clear, Rain, Storm, etc.) for each of the next 7 days — each maps to a typical pressure value under the hood
The app simulates thousands of possible weeks using the fitted arrival and
length-of-stay distributions, adjusting daily intake based on that day's
forecasted pressure
Output: probability of exceeding capacity at least once, average days over
capacity, and the average simulated population trajectory across the week
Limitations

Pressure explains only a small share of day-to-day intake variance (other
factors — season, day-of-week, local events — matter more). This tool should
be read as a modest risk adjustment on top of baseline volume, not a
precise intake forecast.

Tech Stack

Python, NumPy, SciPy (distribution fitting, negative binomial regression),
Streamlit (deployment)

Related Projects
Moon Phase & Weather Analysis — original discovery of the pressure effect
Shelter Return Risk Predictor — companion tool predicting individual animal return risk
