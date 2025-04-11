from commission_structure.commission import calculate_profit_after_commission_binance

buy_price = 2595
buy_amount_usdt = 57.609
sell_price = 2700
order_type_buy = order_type_sell = 'LIMIT'
fee_rate_maker=0.001
fee_rate_taker=0.001

calculate_profit_after_commission_binance(buy_price=buy_price, buy_amount_usdt=buy_amount_usdt,\
                                          sell_price=sell_price, order_type_buy=order_type_buy,\
                                            order_type_sell=order_type_sell, fee_rate_maker=fee_rate_maker,\
                                                fee_rate_taker=fee_rate_taker)