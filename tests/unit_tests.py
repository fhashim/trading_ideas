from commission_structure.commission import calculate_total_commission_binance
from optimizers.optimizer import calculate_profit_binance

import pytest

# Payload with test cases
payload = [
    {
        "input": {
            "buy_price": 59450.0,
            "buy_amount_usdt": 59.45,
            "sell_price": 59000.0,
            "fee_rate_maker": 0.001,
            "fee_rate_taker": 0.001
        },
        "expected_output": -0.56845  # Expected net profit for the first test case
    }
]

# Parametrized test to loop over the payload
@pytest.mark.parametrize("test_case", payload)
def test_calculate_profit_binance(test_case):
    # Unpack the inputs from the payload
    buy_price = test_case['input']['buy_price']
    buy_amount_usdt = test_case['input']['buy_amount_usdt']
    sell_price = test_case['input']['sell_price']
    fee_rate_maker = test_case['input']['fee_rate_maker']
    fee_rate_taker = test_case['input']['fee_rate_taker']
    
    # Call the function with the current test case inputs
    net_profit = calculate_profit_binance(
        buy_price=buy_price,
        buy_amount_usdt=buy_amount_usdt,
        sell_price=sell_price,
        calculate_total_commission_binance=calculate_total_commission_binance,
        fee_rate_maker=fee_rate_maker,
        fee_rate_taker=fee_rate_taker
    )
    
    # Assert the expected output matches the calculated output
    assert pytest.approx(net_profit, rel=1e-6) == test_case['expected_output'], \
        f"Expected {test_case['expected_output']}, got {net_profit}"

# To run the test, you would use pytest in your terminal:
# pytest -v

