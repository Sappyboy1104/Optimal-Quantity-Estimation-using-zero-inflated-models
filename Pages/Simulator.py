import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom
from math import factorial
import warnings

# Suppress runtime warnings that can occur during optimization
warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(layout="wide", page_title="Demand Distribution Fitting")

st.title("Demand Distribution Fitting Tool")


# --- emp_cdf and ks gof ---

def empirical_cdf(data, k_max):
    """Calculates the empirical Cumulative Distribution Function (CDF)."""
    counts = np.bincount(data, minlength=k_max + 1)
    pmf = counts/ len(data)
    return np.cumsum(pmf)


def ks_statistic(empirical_cdf, theoretical_cdf_truncated):
    return np.max(np.abs(empirical_cdf - theoretical_cdf_truncated))


def get_goodness_of_fit(ks_stat, n):
    if n == 0:
        return "N/A"
    # Critical values for KS test: c(alpha) / sqrt(n)
    crit_5_percent = 1.36 / np.sqrt(n)
    crit_1_percent = 1.63 / np.sqrt(n)

    if ks_stat < crit_5_percent:
        return "Strong Fit"
    elif ks_stat < crit_1_percent:
        return "Poor Fit"
    else:
        return "No Fit"


# Model_1 ZISPD
def stuttering_poisson_pmf(k_max, lambd, alphas):
    r = len(alphas)
    k_max = int(k_max)
    probs = np.zeros(k_max + 1)
    if k_max >= 0:
        try:
            probs[0] = np.exp(-lambd)
        except OverflowError:
            return np.zeros(k_max + 1)

    for k in range(1, k_max + 1):
        total = 0
        for j in range(1, min(k, r) + 1):
            total += j * lambd * alphas[j - 1] * probs[k - j]
        if k > 0:
            probs[k] = total / k
    return probs

def zispd_pmf(k_max, lambd, alphas, pi):
    """Calculates the PMF for the ZISPD."""
    spd_probs = stuttering_poisson_pmf(k_max, lambd, alphas)
    zispd_probs = (1 - pi) * spd_probs
    if k_max >= 0:
        zispd_probs[0] = pi + (1 - pi) * spd_probs[0]
    return zispd_probs

def zispd_neg_ll(params, data, r):
    """NLL for ZISPD model, used for MLE fitting."""
    lambd, pi = params[0], params[-1]
    alphas = params[1:r + 1]
    if not (0 <= pi <= 1) or lambd <= 0 or any(a < 0 for a in alphas) or abs(sum(alphas) - 1) > 1e-6:
        return np.inf
    k_max_data = max(data)
    probs = zispd_pmf(k_max_data, lambd, alphas, pi)
    # Add a small epsilon to prevent log(0)
    probs[probs <= 0] = 1e-12
    log_likelihood = sum(np.log(probs[x]) for x in data)
    return -log_likelihood

# Model_2 (ZIP)
def zip_pmf(k_max, lam, pi):
    """Calculates the PMF for the ZIP distribution."""
    probs = (1 - pi) * poisson.pmf(np.arange(k_max + 1), lam)
    if k_max >= 0:
        probs[0] = pi + (1 - pi) * poisson.pmf(0, lam)
    return probs

def zip_neg_ll(params, data):
    """NLL for ZIP model."""
    lam, pi = params
    if lam <= 0 or not (0 <= pi <= 1):
        return np.inf
    log_likelihood = 0
    zero_prob = pi + (1 - pi) * np.exp(-lam)
    for x in data:
        if x == 0:
            prob = zero_prob
        else:
            prob = (1 - pi) * poisson.pmf(x, lam)
        log_likelihood += np.log(prob + 1e-12)
    return -log_likelihood

# Model_3 (ZINB)
def zinb_pmf(k_max, r, p, pi):
    """Calculates the PMF for the ZINB distribution."""
    probs = (1 - pi) * nbinom.pmf(np.arange(k_max + 1), r, p)
    if k_max >= 0:
        probs[0] = pi + (1 - pi) * nbinom.pmf(0, r, p)
    return probs

def zinb_neg_ll(params, data):
    """NLL for ZINB model."""
    r_param, p, pi = params
    if r_param <= 0 or not (0 < p < 1) or not (0 <= pi <= 1):
        return np.inf
    log_likelihood = 0
    zero_prob = pi + (1 - pi) * nbinom.pmf(0, r_param, p)
    for x in data:
        if x == 0:
            prob = zero_prob
        else:
            prob = (1 - pi) * nbinom.pmf(x, r_param, p)
        log_likelihood += np.log(prob + 1e-12)
    return -log_likelihood

# Zero-Inflated Poisson-Geometric (ZIGP / Polya-Aeppli)
def zigp_pmf(k_max, lam, p_geom, pi):
    """Calculates the PMF for the ZIGP (Polya-Aeppli) distribution."""
    pa_probs = np.zeros(k_max + 1)
    if k_max >= 0:
        pa_probs[0] = np.exp(-lam)
    for k in range(1, k_max + 1):
        total = sum(j * p_geom * ((1 - p_geom) ** (j - 1)) * pa_probs[k - j] for j in range(1, k + 1))
        pa_probs[k] = (lam / k) * total
    zigp_probs = (1 - pi) * pa_probs
    if k_max >= 0:
        zigp_probs[0] = pi + (1 - pi) * pa_probs[0]
    return zigp_probs


def zigp_neg_ll(params, data):
    """NLL for ZIGP model."""
    lam, p_geom, pi = params
    if lam <= 0 or not (0 < p_geom < 1) or not (0 <= pi <= 1):
        return np.inf
    k_max = max(data)
    probs = zigp_pmf(k_max, lam, p_geom, pi)
    log_likelihood = sum(np.log(probs[x] + 1e-12) for x in data)
    return -log_likelihood


# --- Model Fitting Engine ---

def fit_models(data, selected_models):
    """Fits the selected models to the data."""
    results = {}
    if not data:
        return results

    k_max_observed = max(data)
    k_max_theoretical = int(max(150, k_max_observed * 3))
    emp_cdf = empirical_cdf(data, k_max_observed)

    # --- Fit NLL-based models (ZIP, ZINB, ZIGP) ---
    if "ZIP" in selected_models:
        res = minimize(zip_neg_ll, [np.mean(data), 0.1], args=(data,), bounds=[(1e-5, None), (0, 1)])
        if res.success:
            params = res.x
            pmf = zip_pmf(k_max_theoretical, *params)
            theoretical_cdf = np.cumsum(pmf)
            ks = ks_statistic(emp_cdf, theoretical_cdf[:len(emp_cdf)])
            results["ZIP"] = {"params": params, "pmf": pmf, "ks": ks, "k": 2}

    if "ZINB" in selected_models:
        mean = np.mean(data)
        var = np.var(data)
        if var > mean:
            p_init = mean / var
            r_init = mean * p_init / (1 - p_init)
        else:
            p_init, r_init = 0.5, 1.0

        res = minimize(zinb_neg_ll, [r_init, p_init, 0.1], args=(data,),
                       bounds=[(1e-5, None), (1e-5, 1 - 1e-5), (0, 1)])
        if res.success:
            params = res.x
            pmf = zinb_pmf(k_max_theoretical, *params)
            theoretical_cdf = np.cumsum(pmf)
            ks = ks_statistic(emp_cdf, theoretical_cdf[:len(emp_cdf)])
            results["ZINB"] = {"params": params, "pmf": pmf, "ks": ks, "k": 3}

    if "ZIGP" in selected_models:
        res = minimize(zigp_neg_ll, [np.mean(data), 0.5, 0.1], args=(data,),
                       bounds=[(1e-5, None), (1e-5, 1 - 1e-5), (0, 1)])
        if res.success:
            params = res.x
            pmf = zigp_pmf(k_max_theoretical, *params)
            theoretical_cdf = np.cumsum(pmf)
            ks = ks_statistic(emp_cdf, theoretical_cdf[:len(emp_cdf)])
            results["ZIGP"] = {"params": params, "pmf": pmf, "ks": ks, "k": 3}


    if "ZISPD" in selected_models:
        best_zispd = {"ks": np.inf}

        scaling_factor = max(1.0, k_max_observed / 150.0)
        data_scaled = np.round(np.array(data) / scaling_factor).astype(int)
        k_max_observed_scaled = max(data_scaled)
        emp_cdf_scaled = empirical_cdf(data_scaled, k_max_observed_scaled)

        for r in range(2, 16):

            init_guess = [np.mean(data_scaled)] + [1 / r] * r + [0.1]
            bounds = [(1e-6, None)] + [(0, 1)] * r + [(0, 1)]
            constraints = {'type': 'eq', 'fun': lambda p: sum(p[1:r + 1]) - 1}

            res = minimize(zispd_neg_ll, init_guess, args=(data_scaled, r),
                           method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 500})
            if res.success:
                fitted_params = res.x
                lambd_scaled, pi = fitted_params[0], fitted_params[-1]
                alphas = fitted_params[1:r + 1]

                k_max_theoretical_scaled = int(max(150, k_max_observed_scaled * 3))
                pmf_scaled = zispd_pmf(k_max_theoretical_scaled, lambd_scaled, alphas, pi)
                theoretical_cdf_scaled = np.cumsum(pmf_scaled)
                ks_val = ks_statistic(emp_cdf_scaled, theoretical_cdf_scaled[:len(emp_cdf_scaled)])

                if ks_val < best_zispd["ks"]:
                    best_zispd = {
                        "params": fitted_params,
                        "pmf": pmf_scaled,
                        "ks": ks_val,
                        "k": 1 + r + 1,
                        "r": r,
                        "scaling_factor": scaling_factor
                    }


        if best_zispd["ks"] != np.inf:
            results["ZISPD"] = best_zispd

    return results


def get_service_levels(pmf, levels, scaling_factor=1.0):
    cdf = np.cumsum(pmf)
    k_max = len(pmf) - 1
    service_quantities = {}
    for level in levels:
        indices = np.where(cdf >= level)[0]
        if len(indices) == 0:
            optimal_quantity = k_max
        else:
            optimal_quantity = indices[0]
        service_quantities[f"{int(level * 100)}%"] = int(np.round(optimal_quantity * scaling_factor))
    return service_quantities


# --- Streamlit UI Layout ---

st.header("1. Input Demand Data")
zero_periods = st.number_input("Number of zero-demand periods", min_value=0, value=15,
                               help="How many periods had exactly zero demand?")
non_zero_input = st.text_area("Non-zero demands (enter the values by separating using comma(,))", "3, 5, 2, 4, 1, 6, 2, 3, 8, 2, 1, 1, 4", height=150,
                              help="Enter all observed non-zero demand values, separated by commas.")

st.header("2. Select Suitable Model to fit")
models_to_fit = []
c1, c2 = st.columns(2)
with c1:
    if st.toggle("ZIP", value=True, help="Zero-Inflated Poisson"): models_to_fit.append("ZIP")
    if st.toggle("ZINB", value=True, help="Zero-Inflated Negative Binomial"): models_to_fit.append("ZINB")
with c2:
    if st.toggle("ZIGP", value=True, help="Zero-Inflated Poisson-Geometric"): models_to_fit.append("ZIGP")
    if st.toggle("ZISPD", value=True, help="Zero-Inflated Stuttering Poisson"): models_to_fit.append("ZISPD")

st.divider()

st.header("3. Analysis Results")
try:
    non_zero_demands = [int(x.strip()) for x in non_zero_input.split(',') if x.strip()]
    demand_array = [0] * zero_periods + non_zero_demands
    n_total = len(demand_array)

    if n_total > 0:
        st.info(f"Total periods analyzed: **{n_total}**. Zero-demand proportion: **{zero_periods / n_total:.2%}**.")

    if n_total > 0 and models_to_fit:
        results = fit_models(demand_array, models_to_fit)

        if not results:
            st.warning("No models could be successfully fitted. Please check your input data or model selection.")
            st.stop()

        # --- Build and Display Summary Table ---
        summary_data = []
        for model, res in results.items():
            params_str = ""
            if model == "ZIP":
                params_str = f"λ={res['params'][0]:.3f}, π={res['params'][1]:.3f}"
            elif model == "ZINB":
                params_str = f"r={res['params'][0]:.3f}, p={res['params'][1]:.3f}, π={res['params'][2]:.3f}"
            elif model == "ZIGP":
                params_str = f"λ={res['params'][0]:.3f}, p_g={res['params'][1]:.3f}, π={res['params'][2]:.3f}"
            elif model == "ZISPD":
                unscaled_lambda = res['params'][0] * res.get('scaling_factor', 1.0)
                params_str = f"r={res['r']}, λ≈{unscaled_lambda:.3f}, π={res['params'][-1]:.3f}"

            summary_data.append({
                "Model": model,
                "KS Statistic": res["ks"],
                "Goodness of Fit": get_goodness_of_fit(res["ks"], n_total),
                "Parameters": params_str,
            })

        summary_df = pd.DataFrame(summary_data).sort_values("KS Statistic").reset_index(drop=True)

        st.subheader("Model Comparison")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


        best_model_name = summary_df.iloc[0]["Model"]
        best_model_results = results[best_model_name]
        best_pmf = best_model_results["pmf"]
        best_model_scaling = best_model_results.get("scaling_factor", 1.0)

        st.subheader(f"  Best Fit Model is: {best_model_name} Model")


        k_max_observed = max(demand_array) if demand_array else 0
        emp_cdf_for_plot = empirical_cdf(demand_array, k_max_observed)
        emp_pmf = np.diff(np.insert(emp_cdf_for_plot, 0, 0))

        best_cdf = np.cumsum(best_pmf)
        plot_limit_candidates = np.where(best_cdf >= 0.999)[0]
        plot_k_max = plot_limit_candidates[0] if len(plot_limit_candidates) > 0 else len(best_pmf) - 1
        plot_k_max = int(max(k_max_observed + 5, plot_k_max))

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.bar(np.arange(k_max_observed + 1), emp_pmf, alpha=0.5, label="Empirical Data", color='gray', width=0.6)

        x_axis_plot = np.arange(plot_k_max + 1)
        for model, result in results.items():
            style = '.-'
            linewidth = 1.5

            model_scaling = result.get("scaling_factor", 1.0)
            if model_scaling > 1.0:
                # Create a sparse representation for the unscaled plot
                unscaled_x = []
                unscaled_pmf_vals = []
                scaled_pmf = result["pmf"]
                for i in range(len(scaled_pmf)):
                    original_x = int(round(i * model_scaling))
                    if original_x <= plot_k_max:
                        unscaled_x.append(original_x)
                        unscaled_pmf_vals.append(scaled_pmf[i])
                ax.plot(unscaled_x, unscaled_pmf_vals, style, label=f"{model} (KS={result['ks']:.4f})", lw=linewidth)

            else:  # No scaling applied
                ax.plot(x_axis_plot, result["pmf"][:len(x_axis_plot)], style, label=f"{model} (KS={result['ks']:.4f})",
                        lw=linewidth)

        ax.set_xlabel("Demand Quantity", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title("Probability Mass Function (PMF) Comparison", fontsize=14)
        ax.set_xlim(left=-0.5, right=plot_k_max + 0.5)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # --- Service Level Calculation ---
        st.subheader("Optimal Quantities for Service Levels")
        st.markdown("Based on the best fitting model, the optimal inventory to hold to meet a given service level is:")

        service_levels = get_service_levels(best_pmf, levels=[0.85, 0.90, 0.95, 0.98, 0.99, 0.995],
                                            scaling_factor=best_model_scaling)

        service_df = pd.DataFrame(list(service_levels.items()), columns=['Service Level', 'Optimal Quantity (Units)'])
        st.dataframe(service_df, use_container_width=True, hide_index=True)

    else:
        st.warning("Please enter some demand data and select at least one model to fit.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Please ensure your demand data is a valid comma-separated list of numbers.")

