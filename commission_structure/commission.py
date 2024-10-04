import requests as r
from decimal import Decimal, ROUND_DOWN

def get_proft_commission_amt_psx(buy_price: float, sell_price: float, \
                                 num_shares: int, day_trade: bool = False):

    """
    Calculate the profit and total cost for a trade on the Pakistan Stock Exchange (PSX) considering commissions, 
    taxes, and handling charges.

    Parameters:
    buy_price (float): The price at which the shares were bought.
    sell_price (float): The price at which the shares are intended to be sold.
    num_shares (int): The number of shares being traded.
    day_trade (bool): A flag indicating whether the trade is a day trade (buy and sell within the same day). 
                      If True, only the buy commission and charges are considered in the total cost.
                      Defaults to False.

    Returns:
    tuple: A tuple containing:
        - profit (float): The net profit after accounting for commissions, taxes, and handling charges.
        - total_cost (float): The total cost incurred for the trade, including commissions, taxes, and handling charges.

    Notes:
    - If the buy or sell price is less than or equal to 4.99, a flat commission equal to the total number of shares multiplied 
      by the price is applied.
    - For prices greater than 4.99, the commission is the higher of a fixed percentage (0.05%) or a variable percentage (0.15%),
      with a tax rate of 13% applied to the commission.
    - Additional costs include CDC handling charges (0.005 per share) and CVT (0.01% of the total buy amount).
    """

    if buy_price <= 4.99:
        buy_commission = num_shares * 0.03
    else:
        buy_commission_fixed = 0.05 * num_shares
        buy_commission_percentage = ((0.15/100) * buy_price) * num_shares
        
        if buy_commission_fixed > buy_commission_percentage:
            buy_commission = buy_commission_fixed
        else:
            buy_commission = buy_commission_percentage

        buy_commission_taxed = buy_commission * 1.15

    if sell_price <= 4.99:
        sell_commission = num_shares * 0.03
    else:
        sell_commission_fixed = 0.05 * num_shares
        sell_commission_percentage = ((0.15/100) * buy_price) * num_shares

        if sell_commission_fixed > sell_commission_percentage:
            sell_commission = sell_commission_fixed
        else:
            sell_commission = sell_commission_percentage

        sell_commission_taxed = sell_commission * 1.15

    buy_cdc_handling_charges = num_shares * 0.005
    sell_cdc_handling_charges = num_shares * 0.005
    buy_cvt = (0.01/100) *  (buy_price * num_shares)

    if day_trade:
        total_cost = buy_commission_taxed + buy_cvt

    else:
        total_cost = buy_commission_taxed + sell_commission_taxed + buy_cdc_handling_charges + sell_cdc_handling_charges

    profit = (sell_price - buy_price) * num_shares - total_cost

    return profit, total_cost



def truncate_to_precision(value: float, pair: str, precision: int = 6) -> float:
    """
    Truncates the given value to the precision specified for the trading pair on Binance.
    
    This function fetches the precision (step size) for the given trading pair from Binance API.
    If the API call is successful, it retrieves the step size and determines the appropriate
    decimal precision for the truncation. The value is then truncated to this precision.

    Parameters:
    ----------
    value : float
        The value to be truncated.
    pair : str
        The trading pair symbol (e.g., 'ETHUSDT').
    precision : int, optional
        The number of decimal places to truncate to, defaults to 6. If found from the
        API, the precision value may be updated based on the pair's step size.

    Returns:
    -------
    float
        The truncated value to the specified precision.
    """
    
    response = r.request(url="https://api.binance.com/api/v3/exchangeInfo", method='GET')

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        symbols = data.get('symbols', [])
    
        # Filter for the trading pair (e.g., 'ETHUSDT')
        pair_info = next((symbol for symbol in symbols if symbol['symbol'] == pair), None)
        
        if pair_info:
            # Extract the step size for this pair, which helps in determining the precision
            step_size = pair_info.get('filters')[1].get('stepSize')
            number_str = str(float(step_size))
            if '.' in number_str:
                precision = len(number_str.split('.')[1])  # Set precision based on the step size
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")

    # Convert the float to Decimal
    decimal_value = Decimal(value)

    # Quantize the Decimal to the specified precision
    truncated_value = decimal_value.quantize(Decimal(f'1.{"0" * precision}'), rounding=ROUND_DOWN)

    return float(truncated_value)

def calculate_profit_after_commission_binance(buy_price: float, buy_amount_usdt: float, sell_price: float,
    order_type_buy: str = 'MARKET', order_type_sell: str = 'MARKET',
    fee_rate_maker: float = 0.001, fee_rate_taker: float = 0.001) -> float:

    if order_type_buy == 'MARKET':
        buy_fee_rate = fee_rate_taker
    else:
        buy_fee_rate = fee_rate_maker

    if order_type_sell == 'MARKET':
        sell_fee_rate = fee_rate_taker
    else:
        sell_fee_rate = fee_rate_maker

    # Calculate the amount of asset bought
    asset_quantity = buy_amount_usdt / buy_price
    
    # Deduct commission on the asset bought
    asset_quantity_available = asset_quantity * (1 - buy_fee_rate)

    asset_quantity_available_sell = truncate_to_precision(asset_quantity_available, 4)
    
    # Selling commission in USDT
    sell_commission = asset_quantity_available_sell * sell_price * sell_fee_rate

    # Profit after sell commission
    profit = asset_quantity_available_sell * sell_price - sell_commission - buy_amount_usdt
    
    return profit


