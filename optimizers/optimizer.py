import numpy as np
from scipy.optimize import minimize
from commission_structure.commission import calculate_profit_after_commission_binance

def calculate_target_selling_stoploss_price_binance(buy_price: float, buy_amount_usdt: float,
                                                    expected_profit: float, loss_tol: float,
                                                    order_type_buy: str, order_type_sell: str,
                                                    calculate_profit_after_commission_binance: callable,
                                                    fee_rate_maker: float = 0.001, fee_rate_taker: float = 0.001) -> tuple:
    """
    Calculate the target selling price and stop-loss price for a trade on Binance based on expected profit and loss tolerance.

    This function uses an optimization process to determine the ideal selling price needed to achieve a target profit, 
    as well as the stop-loss price to limit potential losses, factoring in Binance trading fees.

    Parameters:
    ----------
    buy_price : float
        The price at which the asset was purchased (in USDT).
    buy_amount_usdt : float
        The total amount of USDT used to purchase the asset.
    expected_profit : float
        The desired profit in USDT to achieve from the sale of the asset.
    loss_tol : float
        The acceptable loss in USDT you are willing to tolerate before selling at a stop-loss.
    order_type_buy : str
        The type of buy order used (e.g., "LIMIT" or "MARKET").
    order_type_sell : str
        The type of sell order to be placed (e.g., "LIMIT" or "MARKET").
    calculate_profit_after_commission_binance : callable
        A function to calculate the net profit after considering Binance trading fees (maker/taker).
    fee_rate_maker : float, optional
        The maker fee rate (default is 0.1% or 0.001).
    fee_rate_taker : float, optional
        The taker fee rate (default is 0.1% or 0.001).

    Returns:
    -------
    tuple
        A tuple containing:
        - sell_price (float): The calculated selling price required to achieve the expected profit.
        - stoploss_price (float): The stop-loss price to limit the loss, rounded to two decimal places.

    Notes:
    -----
    - The function optimizes the selling and stop-loss prices that align with the expected profit and loss tolerance using 
      optimization techniques.
    - The `calculate_profit_after_commission_binance` function is applied within the optimization process to evaluate the 
      net profit based on the calculated prices.
    - `fee_rate_maker` and `fee_rate_taker` correspond to the trading fee rates for maker and taker orders, respectively.
    - The initial guess for optimization is calculated based on adjusting the buy price for the desired expected profit.
    """

        
    # Objective function: difference between calculated profit and target profit
    def objective_sell_price(sell_price):
        profit = calculate_profit_after_commission_binance(buy_price=buy_price, buy_amount_usdt=buy_amount_usdt,\
                                          sell_price=sell_price, order_type_buy=order_type_buy,\
                                            order_type_sell=order_type_sell, fee_rate_maker=fee_rate_maker,\
                                                fee_rate_taker=fee_rate_taker)
        return np.abs(profit - expected_profit)
    
    # Objective function: difference between calculated profit and loss tolerance
    def objective_stop_loss_price(sell_price):
        profit = calculate_profit_after_commission_binance(buy_price=buy_price, buy_amount_usdt=buy_amount_usdt,\
                                          sell_price=sell_price, order_type_buy=order_type_buy,\
                                            order_type_sell=order_type_sell, fee_rate_maker=fee_rate_maker,\
                                                fee_rate_taker=fee_rate_taker)
        return np.abs(profit - loss_tol)
    
    # Initial guess for selling price (starting point for the optimizer)
    initial_guess = buy_price * (1 + expected_profit / buy_amount_usdt)
    
    
    # Run the optimizer
    result_sell_price = minimize(objective_sell_price, initial_guess, bounds=[(None, None)])
    
    # Return the optimized selling price
    sell_price = result_sell_price.x[0]

    # Run the optimizer
    result_stop_loss_price = minimize(objective_stop_loss_price, initial_guess, bounds=[(None, None)])
    
    # Return the stop loss price
    stoploss_price = result_stop_loss_price.x[0]

    return np.round(sell_price), np.round(stoploss_price,2)





buy_price = 2595
buy_amount_usdt = 57.609
buy_qty = buy_amount_usdt / buy_price
fee_rate_maker=0.001
fee_rate_taker=0.001


buy_price = 2600
buy_amount_usdt = 1000

calculate_target_selling_stoploss_price_binance(buy_price=buy_price, buy_amount_usdt=buy_amount_usdt,\
                                                expected_profit = 20, loss_tol = 10, \
                                                    order_type_buy = 'LIMIT', order_type_sell = 'LIMIT',\
                                                        calculate_profit_after_commission_binance = \
                                                            calculate_profit_after_commission_binance,\
                                                                fee_rate_maker = 0.001, fee_rate_taker = 0.001)


