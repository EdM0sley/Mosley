import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Black-Scholes functions ---

def black_scholes(S, K, T, r, vol, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")
    return price

def d1_func(S, K, T, r, vol):
    return (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))

def delta(S, K, T, r, vol, option_type='call'):
    d1 = d1_func(S, K, T, r, vol)
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type")

def gamma(S, K, T, r, vol):
    d1 = d1_func(S, K, T, r, vol)
    return norm.pdf(d1) / (S * vol * np.sqrt(T))

def vega(S, K, T, r, vol):
    d1 = d1_func(S, K, T, r, vol)
    return S * norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, vol, option_type='call'):
    d1 = d1_func(S, K, T, r, vol)
    d2 = d1 - vol * np.sqrt(T)
    first_term = - (S * norm.pdf(d1) * vol) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        second_term = - r * K * np.exp(-r * T) * norm.cdf(d2)
        return first_term + second_term
    elif option_type == 'put':
        second_term = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return first_term + second_term
    else:
        raise ValueError("Invalid option type")

def rho(S, K, T, r, vol, option_type='call'):
    d1 = d1_func(S, K, T, r, vol)
    d2 = d1 - vol * np.sqrt(T)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type")

# --- Plotting functions ---

def plot_option_prices_vs_strike(S, K_values, T, r, sigma):
    call_prices = black_scholes(S, K_values, T, r, sigma, 'call')
    put_prices = black_scholes(S, K_values, T, r, sigma, 'put')

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(K_values, call_prices, label="Call Option Price", color='blue')
    ax.plot(K_values, put_prices, label="Put Option Price", color='red')
    ax.axvline(S, color='gray', linestyle='--', label="Current Stock Price (S)")
    ax.set_title("Option Prices vs Strike Price")
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Option Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_option_prices_vs_time(S, K, T_values, r, sigma):
    call_prices_T = black_scholes(S, K, T_values, r, sigma, 'call')
    put_prices_T = black_scholes(S, K, T_values, r, sigma, 'put')

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(T_values, call_prices_T, label="Call Option Price", color='blue')
    ax.plot(T_values, put_prices_T, label="Put Option Price", color='red')
    ax.set_title("Option Prices vs Time to Maturity")
    ax.set_xlabel("Time to Maturity (T in years)")
    ax.set_ylabel("Option Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_greeks_vs_strike(S, K_values, T, r, sigma):
    call_delta = delta(S, K_values, T, r, sigma, 'call')
    put_delta = delta(S, K_values, T, r, sigma, 'put')
    call_gamma = gamma(S, K_values, T, r, sigma)
    call_vega = vega(S, K_values, T, r, sigma)
    call_theta = theta(S, K_values, T, r, sigma, 'call')
    put_theta = theta(S, K_values, T, r, sigma, 'put')
    call_rho = rho(S, K_values, T, r, sigma, 'call')
    put_rho = rho(S, K_values, T, r, sigma, 'put')

    fig, axs = plt.subplots(5, 1, figsize=(10, 22))

    axs[0].plot(K_values, call_delta, label="Call Delta", color='blue')
    axs[0].plot(K_values, put_delta, label="Put Delta", color='red')
    axs[0].axvline(S, color='gray', linestyle='--')
    axs[0].set_title("Delta vs Strike Price")
    axs[0].set_xlabel("Strike Price (K)")
    axs[0].set_ylabel("Delta")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(K_values, call_gamma, label="Gamma (Call & Put)", color='green')
    axs[1].axvline(S, color='gray', linestyle='--')
    axs[1].set_title("Gamma vs Strike Price")
    axs[1].set_xlabel("Strike Price (K)")
    axs[1].set_ylabel("Gamma")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(K_values, call_vega, label="Vega (Call & Put)", color='purple')
    axs[2].axvline(S, color='gray', linestyle='--')
    axs[2].set_title("Vega vs Strike Price")
    axs[2].set_xlabel("Strike Price (K)")
    axs[2].set_ylabel("Vega")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(K_values, call_theta, label="Call Theta", color='orange')
    axs[3].plot(K_values, put_theta, label="Put Theta", color='brown')
    axs[3].axvline(S, color='gray', linestyle='--')
    axs[3].set_title("Theta vs Strike Price")
    axs[3].set_xlabel("Strike Price (K)")
    axs[3].set_ylabel("Theta")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(K_values, call_rho, label="Call Rho", color='cyan')
    axs[4].plot(K_values, put_rho, label="Put Rho", color='magenta')
    axs[4].axvline(S, color='gray', linestyle='--')
    axs[4].set_title("Rho vs Strike Price")
    axs[4].set_xlabel("Strike Price (K)")
    axs[4].set_ylabel("Rho")
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

def plot_greeks_vs_time(S, K, T_values, r, sigma):
    call_delta_T = delta(S, K, T_values, r, sigma, 'call')
    put_delta_T = delta(S, K, T_values, r, sigma, 'put')
    call_gamma_T = gamma(S, K, T_values, r, sigma)
    call_vega_T = vega(S, K, T_values, r, sigma)
    call_theta_T = theta(S, K, T_values, r, sigma, 'call')
    put_theta_T = theta(S, K, T_values, r, sigma, 'put')
    call_rho_T = rho(S, K, T_values, r, sigma, 'call')
    put_rho_T = rho(S, K, T_values, r, sigma, 'put')

    fig, axs = plt.subplots(5, 1, figsize=(10, 22))

    axs[0].plot(T_values, call_delta_T, label="Call Delta", color='blue')
    axs[0].plot(T_values, put_delta_T, label="Put Delta", color='red')
    axs[0].set_title("Delta vs Time to Maturity")
    axs[0].set_xlabel("Time to Maturity (T in years)")
    axs[0].set_ylabel("Delta")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(T_values, call_gamma_T, label="Gamma (Call & Put)", color='green')
    axs[1].set_title("Gamma vs Time to Maturity")
    axs[1].set_xlabel("Time to Maturity (T in years)")
    axs[1].set_ylabel("Gamma")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(T_values, call_vega_T, label="Vega (Call & Put)", color='purple')
    axs[2].set_title("Vega vs Time to Maturity")
    axs[2].set_xlabel("Time to Maturity (T in years)")
    axs[2].set_ylabel("Vega")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(T_values, call_theta_T, label="Call Theta", color='orange')
    axs[3].plot(T_values, put_theta_T, label="Put Theta", color='brown')
    axs[3].set_title("Theta vs Time to Maturity")
    axs[3].set_xlabel("Time to Maturity (T in years)")
    axs[3].set_ylabel("Theta")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(T_values, call_rho_T, label="Call Rho", color='cyan')
    axs[4].plot(T_values, put_rho_T, label="Put Rho", color='magenta')
    axs[4].set_title("Rho vs Time to Maturity")
    axs[4].set_xlabel("Time to Maturity (T in years)")
    axs[4].set_ylabel("Rho")
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# --- Main Streamlit app ---

def main():
    st.title("Black-Scholes Option Pricer with Greeks (including Rho)")

    # User inputs with defaults resembling Netflix example
    S = st.number_input("Current stock price (S)", value=1177.75, min_value=0.01)
    r = st.number_input("Risk-free interest rate (r, decimal)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f")
    sigma = st.number_input("Volatility (sigma, decimal)", value=0.254, min_value=0.0, max_value=5.0, format="%.4f")
    T = st.slider("Time to maturity (T, years)", 0.01, 2.0, 0.25, step=0.01)
    # Dynamic strike range slider centered on S:
    K_min = max(0.01, S * 0.5)
    K_max = S * 1.5
    K = st.slider("Strike price (K)", K_min, K_max, float(S), step=0.01)

    # Generate strike and time ranges for plots
    K_values = np.linspace(K_min, K_max, 100)
    T_values = np.linspace(0.01, 2, 100)

    # Display current input summary
    st.write(f"### Inputs:")
    st.write(f"- Stock Price (S): {S}")
    st.write(f"- Strike Price (K): {K}")
    st.write(f"- Time to Maturity (T): {T} years")
    st.write(f"- Risk-free Rate (r): {r*100:.2f}%")
    st.write(f"- Volatility (sigma): {sigma*100:.2f}%")

    # Compute and display option prices for the single strike/time inputs
    call_price = black_scholes(S, K, T, r, sigma, 'call')
    put_price = black_scholes(S, K, T, r, sigma, 'put')
    st.write(f"### Option Prices at Strike {K} and Time {T}:")
    st.write(f"- Call Option Price: ${call_price:.2f}")
    st.write(f"- Put Option Price: ${put_price:.2f}")

    # Plotting
    st.write("## Option Prices vs Strike Price")
    plot_option_prices_vs_strike(S, K_values, T, r, sigma)

    st.write("## Option Prices vs Time to Maturity")
    plot_option_prices_vs_time(S, K, T_values, r, sigma)

    st.write("## Greeks vs Strike Price")
    plot_greeks_vs_strike(S, K_values, T, r, sigma)

    st.write("## Greeks vs Time to Maturity")
    plot_greeks_vs_time(S, K, T_values, r, sigma)

if __name__ == "__main__":
    main()