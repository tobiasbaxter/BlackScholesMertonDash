# Black-Scholes-Merton Model Dashboard for European Option Pricing

This interactive web application, built with Plotly Dash, provides a comprehensive tool for pricing European options using the Black-Scholes-Merton (BSM) model. Users can specify an underlying asset, strike price, expiration date, and risk-free rate to calculate option prices and their corresponding Greeks.

---

## Methodology Overview

The dashboard guides the user through the process of pricing European options:

* **Data Retrieval & Volatility Calculation:** The user inputs an equity ticker. The application then retrieves historical daily close prices for the selected ticker (last 2 years) using `yfinance`. It calculates the annualized volatility of the underlying asset using the standard deviation of its daily logarithmic returns. The most recent price and calculated volatility are displayed.

* **Black-Scholes-Merton Model:** The core of the dashboard is the Black-Scholes-Merton model implementation. The user provides:
    * **Strike Price (K):** The price at which the option holder can buy (call) or sell (put) the underlying asset.
    * **Expiration Date:** The date on which the option contract expires. This is used to calculate the **time to expiration (T)** in years.
    * **Risk-Free Rate (r):** The theoretical rate of return of an investment with zero risk. By default, the dashboard fetches the current 10-Year U.S. Treasury yield as a proxy.

* **Option Price Calculation:** Using the inputs (**Underlying Price (S)**, **Strike Price (K)**, **Time to Expiration (T)**, **Risk-Free Rate (r)**, and **Volatility (σ)**), the dashboard calculates the theoretical price for both European **call** and **put** options based on the BSM formula.

* **Greeks Calculation:** The dashboard also computes and displays the "**Greeks**," which are measures of an option's sensitivity to various factors:
    * **Delta:** Sensitivity of the option price to a change in the underlying asset's price ($\frac{\partial V}{\partial S}$).
    * **Gamma:** Sensitivity of Delta to a change in the underlying asset's price ($\frac{\partial^2 V}{\partial S^2}$).
    * **Vega:** Sensitivity of the option price to a change in the underlying asset's volatility ($\frac{\partial V}{\partial \sigma}$).
    * **Theta:** Sensitivity of the option price to the passage of time (time decay) ($\frac{\partial V}{\partial t}$).
    * **Rho:** Sensitivity of the option price to a change in the risk-free interest rate ($\frac{\partial V}{\partial r}$).

---

## How to Use the App
1.  Navigate to the live app: #placeholder#
2.  **Enter a Ticker:** Type the ticker symbol of the desired underlying asset (e.g., AMZN, AAPL, GOOG, MSFT) into the input field and press Enter. The dashboard will display the historical price chart, the most recent price, and the calculated annualized volatility.
3.  **Specify Option Parameters:**
    * **Strike Price:** Enter your desired strike price for the option.
    * **Expiration:** Select an expiration date using the date picker.
    * **Risk-Free Rate:** Optionally, enter a custom risk-free interest rate (in percentage). If left blank, the dashboard will use the current 10-Year U.S. Treasury yield.
4.  **Review Results:** The dashboard will automatically update to display:
    * The input parameters used for the BSM calculation.
    * A table showing the calculated call and put option prices.
    * A table presenting the values of the Greeks (Delta, Gamma, Vega, Theta, and Rho) for both call and put options.

---

## Technical Stack

* **Backend & Web Framework:** Dash, Flask, `yfinance`
* **Data Manipulation:** Pandas, NumPy
* **Statistical Functions:** `scipy.stats` (for cumulative distribution function and probability density function)
* **Plotting:** Plotly Express
* **Deployment:** Render

---

## Limitations and Further Analyses

* **European Options Only:** This dashboard currently supports only European options, which can be exercised only at expiration. Future enhancements could include support for American options, which can be exercised at any time up to expiration, requiring more complex pricing models (e.g., binomial tree models or Monte Carlo simulations).
* **BSM Model Assumptions:** The Black-Scholes-Merton model relies on several simplifying assumptions (e.g., constant volatility, no dividends, no transaction costs, log-normally distributed returns). In reality, these assumptions may not hold, leading to discrepancies between theoretical and actual option prices.
* **Volatility Proxy:** The dashboard uses historical volatility calculated from past price data. Implied volatility, derived from market prices of options, often provides a more forward-looking measure and could be incorporated for more realistic pricing.
* **Live Data:** The current implementation fetches historical data up to the previous day. For real-time pricing, integrating a live data feed for underlying prices and interest rates would be beneficial.
* **Additional Greeks:** While the core Greeks are included, further extensions could incorporate other sensitivity measures like Vanna, Charm, or Speed.
* **Visualizations of Greeks:** Plotting how the Greeks change with variations in underlying price, time to expiry, or volatility would offer deeper insights into option behavior.
* **Model Calibration:** Implementing methods to calibrate the BSM model to observed market prices (e.g., by finding the implied volatility) could enhance its practical utility.