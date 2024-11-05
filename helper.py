def padd_trade_map(addr_to_trades, padd_length=80):
    # Iterate over each address in addr_to_trades
    for addr, trades in addr_to_trades.items():
        # Check the current length of the trades list
        if len(trades) < padd_length:
            # Pad with scalar 0 until the length is padd_length
            trades.extend([0] * (padd_length - len(trades)))
        elif len(trades) > padd_length:
            # Trim the list to the first padd_length elements
            addr_to_trades[addr] = trades[:padd_length]

    return addr_to_trades
