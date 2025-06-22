# Black-Scholes-Merton model dashboard for pricing European option

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import yfinance as yf
from datetime import date, datetime, timedelta
import scipy.stats as si

class BlackScholesMerton:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Underlying asset price
        self.K = K        # Option strike price
        self.T = T        # Time to expiration in years
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self):
        return self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)
    
    def put_option_price(self):
        return self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0)
        
class BSMGreeks(BlackScholesMerton):
    def delta_call(self):
        return si.norm.cdf(self.d1(), 0.0, 1.0)
    
    def delta_put(self):
        return si.norm.cdf(self.d1(), 0.0, 1.0) - 1
    
    def gamma(self):
        return si.norm.pdf(self.d1(), 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        return self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * np.sqrt(self.T)
    
    def theta_call(self):
        return - (self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)
    
    def theta_put(self):
        return - (self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0)
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)
    
    def rho_put(self):
        return - self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0)

def format_val(value, precision=3):
    return f"{value:.{precision}f}" if not np.isnan(value) else "N/A"
    

app = Dash(__name__)
server = app.server

colors = {
    'background': "#FFFFFF",
    'text': "#000000"
}

# ---App Layout---
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    # Ticker Input
    html.H3("Type ticker of underlying asset and press Enter:",
        style={'textAlign': 'center'}),
    html.Div(style={'textAlign': 'center'}, children=[
        dcc.Input(id='ticker-input', debounce=True, type='text', value='AMZN',
            style={'textAlign': 'center'})
    ]),

    html.H1(
        id='dash-title',
        style={'textAlign': 'center'}
    ),

    html.Div(
        id='description-text',
        style={'textAlign': 'center'}
    ),

    dcc.Graph(id='price-chart'),

    html.H3('Most Recent Price:',
            style={'textAlign': 'center'}
    ),

    html.H3(id='recent-price',
            style={'textAlign': 'center'}
    ),

    html.H3('Annualised Volatility:',
            style={'textAlign': 'center'}
    ),

    html.H3(id='vol',
            style={'textAlign': 'center'}
    ),
    
    # Option parameter inputs
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'flexWrap': 'wrap'}, # Using flexbox for alignment
        children=[
            html.Div([
                html.H4("Strike Price:", style={'marginBottom': '5px'}),
                dcc.Input(id = 'strike', debounce=True, type='number', placeholder='e.g., 150', style={'width': '150px'})
            ], style={'textAlign': 'center', 'flexGrow': '1', 'minWidth': '200px'}), # Adjust minWidth as needed

            html.Div([
                html.H4("Expiration:", style={'marginBottom': '5px'}),
                dcc.DatePickerSingle(
                    id='expiration',
                    initial_visible_month=date.today(), # type: ignore
                    date=date.today() + timedelta(days=30), # type: ignore
                    style={'width': '150px'} # Adjust width of date picker
                ),
            ], style={'textAlign': 'center', 'flexGrow': '1', 'minWidth': '200px'}),

            html.Div([
                html.H4("Risk-Free Rate % (Default: Current 10Y T-Note):", style={'marginBottom': '5px'}),
                dcc.Input(id = 'risk-free-rate', debounce=True, type='number', placeholder='e.g., 4.5', style={'width': '150px'})
            ], style={'textAlign': 'center', 'flexGrow': '1', 'minWidth': '200px'})
        ]
    ),

    # BSM outputs
    html.H3("Black-Scholes-Merton Inputs and Outputs (In Table):",
        style={'textAlign': 'center'}),
    html.Div(children=[
        html.Pre(id='BSM-result')
        ],
        style={'textAlign': 'center'}),
    html.Div(children=[
        html.Pre(id='BSM-table', style={'width': '80%', 'margin': 'auto'})
        ],
        style={'textAlign': 'center'})

])

# --- Callback to Update the Graphs and Text ---
@callback(
    Output('price-chart', 'figure'),
    Output('dash-title', 'children'),
    Output('description-text', 'children'),
    Output('recent-price', 'children'),
    Output('vol', 'children'),
    Output('BSM-result', 'children'),
    Output('BSM-table', 'children'),
    Input('ticker-input', 'value'),
    Input('strike', 'value'),
    Input('expiration', 'date'),
    Input('risk-free-rate', 'value'),
    
    )

def update_dash(input_ticker, strike_price, exp_date, riskfree):
    #################################
    # --- Default Dashboard Inputs + Ticker Selection---
    initial_ticker = 'AMZN' # Default asset
    ticker = str(input_ticker).upper().strip() if input_ticker else initial_ticker
    bond_10yr = yf.Ticker("^TNX").history(period='1d')
    rfr = bond_10yr['Close'].iloc[-1]/100 # Default risk-free rate

    # Title and description
    dash_title = (f"Option Pricing for: {ticker}")
    description_text = (
        f"This dashboard displays the daily closing price and annualised volatility for a selected underlying asset. "
        "It shows the historical price movements and calculates market volatility, using standard deviation of log returns as a proxy for unobservable volatility. "
        "The dashboard uses these parameters, along with a specified strike price, expiration date, "
        "and the prevailing risk-free interest rate (defaulting to the current 10-Year U.S. Treasury yield), "
        "to calculate European call and put option prices using the Black-Scholes-Merton model. "
        "Additionally, it computes and displays the 'Greeks' (Delta, Gamma, Vega, Theta, and Rho), "
        "which provide information on the sensitivity of option prices to various market factors."
    )

    #################################
    # --- Ticker Data Retrieval (Last 2Y) ---
    prices = yf.download(ticker, start = datetime.today() - pd.Timedelta(days= 1 + 365 * 2) , end = datetime.today() - pd.Timedelta(days=1))
    if prices.empty: # type: ignore
        # If not, return an empty chart and an. error message
        error_title = f"Ticker '{ticker}' not found"
        error_desc = "Please enter a valid ticker."
        # Return empty outputs for all 16 outputs
        return {}, dash_title, description_text, {}, {}, {}
    
    #################################
    # --- Price Chart ---
    # Price chart
    price_chart = px.line(
        x = prices.index, #type: ignore
        y = prices['Close'].values.flatten(), #type: ignore
        title = f"{ticker} Daily Close Price"
    )
    price_chart.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Price"
    )
    price_chart.update_xaxes(rangeslider_visible=True)

    recent_price = round(prices['Close'].iloc[-1], 3) # type: ignore

    # Volatility
    log_returns = (np.log(prices['Close']).diff()).dropna() # type: ignore
    volatility = (log_returns.std() * np.sqrt(252)).iloc[0]
    display_vol = f'{format_val(volatility*100)}%'

    # Black-Scholes-Merton implementation
    S = prices['Close'].iloc[-1][ticker] # type: ignore
    K = strike_price if strike_price is not None else S
    T_days = (date.fromisoformat(exp_date) - date.today()).days # Calculate time to expiration in days
    T = T_days / 365.25 if T_days > 0 else 0.001 # Convert to years, ensure T is not zero for log and sqrt
    r = riskfree / 100 if riskfree is not None else rfr
    sigma = volatility

    # Handle cases where BSM inputs might be invalid for calculation
    if K is None or K <= 0 or T <= 0 or sigma <= 0:
        bsm_result_text = "Please ensure Strike Price is entered, Expiration is in the future, and Volatility is positive."
        return price_chart, dash_title, description_text, recent_price, display_vol, bsm_result_text
    
    bsm = BlackScholesMerton(S=S, K=K, T=T, r=r, sigma=sigma)
    call_price = bsm.call_option_price()
    put_price = bsm.put_option_price()
    
    # Format the output for display
    bsm_result_text = (
        f"Input Parameters Below:\n"
        f"Underlying Price (S): {format_val(S)}\n"
        f"Strike Price (K): {format_val(K)}\n"
        f"Time to Expiration (T): {format_val(T)} years\n"
        f"Risk-Free Rate (r): {format_val(r*100)}%\n"
        f"Volatility (σ): {format_val(sigma*100)}%\n\n"
    )

    bsm_greeks = BSMGreeks(S=S, K=K, T=T, r=r, sigma=sigma)
    delta_c = bsm_greeks.delta_call()
    delta_p = bsm_greeks.delta_put()
    gamma = bsm_greeks.gamma()
    vega = bsm_greeks.vega()
    theta_c = bsm_greeks.theta_call()
    theta_p = bsm_greeks.theta_put()
    rho_c = bsm_greeks.rho_call()
    rho_p = bsm_greeks.rho_put()

    # Format the output for display in an HTML table
    bsm_result_table = html.Table(
        style={'width': '100%', 'borderCollapse': 'collapse', 'textAlign': 'center'},
        children=[
            html.Thead(children=[
                html.Tr(children=[
                    html.Th(style={'border': '1px solid black', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                    html.Th('Call', colSpan=1, style={'border': '1px solid black', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                    html.Th('Put', colSpan=1, style={'border': '1px solid black', 'padding': '8px', 'backgroundColor': '#f2f2f2'})
                ]),
                html.Tr(children=[
                    html.Th('Price', style={'border': '1px solid black', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                    html.Td(f"{format_val(call_price)}", style={'border': '1px solid black', 'padding': '8px'}),
                    html.Td(f"{format_val(put_price)}", style={'border': '1px solid black', 'padding': '8px'})
                ])
            ]),
            html.Tbody(children=[
                html.Tr(children=[
                    html.Td(children=[html.B('Delta'), html.Div('∂V/∂S')], style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'left'}),
                    html.Td(f"{format_val(delta_c)}", style={'border': '1px solid black', 'padding': '8px'}),
                    html.Td(f"{format_val(delta_p)}", style={'border': '1px solid black', 'padding': '8px'})
                ]),
                html.Tr(children=[
                    html.Td(children=[html.B('Gamma'), html.Div('∂²V/∂S²')], style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'left'}),
                    html.Td(f"{format_val(gamma)}", colSpan=2, style={'border': '1px solid black', 'padding': '8px'}) # Merged for Gamma
                ]),
                html.Tr(children=[
                    html.Td(children=[html.B('Vega'), html.Div('∂V/∂σ')], style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'left'}),
                    html.Td(f"{format_val(vega)}", colSpan=2, style={'border': '1px solid black', 'padding': '8px'}) # Merged for Vega
                ]),
                html.Tr(children=[
                    html.Td(children=[html.B('Theta'), html.Div('∂V/∂t')], style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'left'}),
                    html.Td(f"{format_val(theta_c)}", style={'border': '1px solid black', 'padding': '8px'}),
                    html.Td(f"{format_val(theta_p)}", style={'border': '1px solid black', 'padding': '8px'})
                ]),
                html.Tr(children=[
                    html.Td(children=[html.B('Rho'), html.Div('∂V/∂r')], style={'border': '1px solid black', 'padding': '8px', 'textAlign': 'left'}),
                    html.Td(f"{format_val(rho_c)}", style={'border': '1px solid black', 'padding': '8px'}),
                    html.Td(f"{format_val(rho_p)}", style={'border': '1px solid black', 'padding': '8px'})
                ])
            ])
        ]
    )

    return price_chart, dash_title, description_text, recent_price, display_vol, bsm_result_text, bsm_result_table

if __name__ == '__main__':
    app.run(debug=True)