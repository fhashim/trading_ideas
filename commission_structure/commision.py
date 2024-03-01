
def get_commission_amt(buy_price, sell_price, num_shares, day_trade=False):

    if buy_price <= 4.99:
        buy_commission = num_shares * buy_price
    else:
        buy_commission_fixed = 0.05 * buy_price * num_shares
        buy_commission_percentage = ((0.15/100) * buy_price) * num_shares
        
        if buy_commission_fixed > buy_commission_percentage:
            buy_commission = buy_commission_fixed
        else:
            buy_commission = buy_commission_percentage

        buy_commission_taxed = buy_commission * 1.13

    if sell_price <= 4.99:
        sell_commission = num_shares * buy_price
    else:
        sell_commission_fixed = 0.05 * buy_price * num_shares
        sell_commission_percentage = ((0.15/100) * buy_price) * num_shares

        if sell_commission_fixed > sell_commission_percentage:
            sell_commission = sell_commission_fixed
        else:
            sell_commission = sell_commission_percentage

        sell_commission_taxed = sell_commission * 1.13

    buy_cdc_handling_charges = num_shares * 0.005
    sell_cdc_handling_charges = num_shares * 0.005
    buy_cvt = (0.01/100) *  (buy_price * num_shares)

    if day_trade:
        total_cost = buy_commission_taxed + buy_cvt

    else:
        total_cost = buy_commission_taxed + sell_commission_taxed + buy_cdc_handling_charges + sell_cdc_handling_charges

    profit = (sell_price - buy_price) * num_shares - total_cost

    return profit, total_cost

buy_price = 154.51
sell_price = 160
num_shares = 100
day_trade = True
profit, total_cost = get_commission_amt(154.51, 160, 100, True)