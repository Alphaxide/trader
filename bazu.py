import ccxt.pro as ccxt
import asyncio
import resend
from datetime import datetime, timedelta
import statistics
import json
import os

class OKXVolatilityBot:
    def __init__(self, api_key, api_secret, passphrase, resend_key, target_email, testnet=False):
        # --- API Credentials ---
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        # Initialize Exchange
        self.exchange = ccxt.okx({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("🔧 Running in TESTNET mode")

        # --- Email Configuration ---
        resend.api_key = resend_key
        self.receiver_email = target_email
        self.sender_email = "onboarding@resend.dev"

        # --- Strategy Settings ---
        self.profit_target = 0.03   # 3% Profit
        self.stop_loss = 0.03       # 3% Stop Loss
        self.risk_per_trade = 0.98  # Use 98% of balance
        
        # --- Watchlist Settings ---
        self.watchlist = []
        self.last_scan_time = datetime.min
        self.scan_interval = 2  # Rescan every 2 hours
        self.num_coins = 20     # Track top 20 mid-tier volatile coins
        
        # --- State ---
        self.running = False
        self.positions = {}
        self.initial_balance = 0
        self.trade_history = []
        
        # --- Persistence Settings (for Render restarts) ---
        self.state_file = "bot_state.json"
        self.load_state()  # Load previous state on startup

    def send_notification(self, subject, message):
        """Sends email via Resend"""
        try:
            resend.Emails.send({
                "from": self.sender_email,
                "to": [self.receiver_email],
                "subject": subject,
                "html": message,
            })
            print(f"📧 Email Sent: {subject}")
        except Exception as e:
            print(f"❌ Email error: {e}")

    def save_state(self):
        """Saves bot state to JSON file for persistence across restarts"""
        try:
            # Convert datetime objects to strings for JSON serialization
            state = {
                'positions': {},
                'trade_history': [],
                'initial_balance': self.initial_balance,
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time != datetime.min else None,
                'watchlist': self.watchlist
            }
            
            # Serialize positions
            for pos_id, pos in self.positions.items():
                state['positions'][pos_id] = {
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'take_profit': pos['take_profit'],
                    'stop_loss': pos['stop_loss'],
                    'entry_time': pos['entry_time'].isoformat(),
                    'rsi_entry': pos['rsi_entry'],
                    'position_in_range': pos['position_in_range']
                }
            
            # Serialize trade history
            for trade in self.trade_history:
                state['trade_history'].append({
                    'symbol': trade['symbol'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'quantity': trade['quantity'],
                    'pnl_pct': trade['pnl_pct'],
                    'pnl_usd': trade['pnl_usd'],
                    'reason': trade['reason'],
                    'entry_time': trade['entry_time'].isoformat(),
                    'exit_time': trade['exit_time'].isoformat(),
                    'duration': str(trade['duration']),
                    'balance_after': trade['balance_after']
                })
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print("💾 State saved successfully")
        except Exception as e:
            print(f"❌ Failed to save state: {e}")

    def load_state(self):
        """Loads bot state from JSON file if it exists"""
        try:
            if not os.path.exists(self.state_file):
                print("📝 No previous state found - starting fresh")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore positions
            for pos_id, pos in state.get('positions', {}).items():
                self.positions[pos_id] = {
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'take_profit': pos['take_profit'],
                    'stop_loss': pos['stop_loss'],
                    'entry_time': datetime.fromisoformat(pos['entry_time']),
                    'rsi_entry': pos['rsi_entry'],
                    'position_in_range': pos['position_in_range']
                }
            
            # Restore trade history
            for trade in state.get('trade_history', []):
                duration_parts = trade['duration'].split(':')
                self.trade_history.append({
                    'symbol': trade['symbol'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'quantity': trade['quantity'],
                    'pnl_pct': trade['pnl_pct'],
                    'pnl_usd': trade['pnl_usd'],
                    'reason': trade['reason'],
                    'entry_time': datetime.fromisoformat(trade['entry_time']),
                    'exit_time': datetime.fromisoformat(trade['exit_time']),
                    'duration': timedelta(hours=int(duration_parts[0]), 
                                         minutes=int(duration_parts[1]), 
                                         seconds=int(duration_parts[2].split('.')[0])),
                    'balance_after': trade['balance_after']
                })
            
            # Restore other state
            self.initial_balance = state.get('initial_balance', 0)
            self.watchlist = state.get('watchlist', [])
            
            if state.get('last_scan_time'):
                self.last_scan_time = datetime.fromisoformat(state['last_scan_time'])
            
            if self.positions:
                print(f"🔄 RESUMED: Found {len(self.positions)} ongoing trade(s)")
                for pos_id, pos in self.positions.items():
                    print(f"   └─ {pos['symbol']} opened at ${pos['entry_price']:.6f}")
                    
                # Send resume notification
                resume_msg = "<h2>🔄 Bot Resumed After Restart</h2>"
                resume_msg += f"<p><strong>Active Positions:</strong> {len(self.positions)}</p>"
                resume_msg += "<ul>"
                for pos in self.positions.values():
                    resume_msg += f"<li><strong>{pos['symbol']}</strong> - Entry: ${pos['entry_price']:.6f} | TP: ${pos['take_profit']:.6f} | SL: ${pos['stop_loss']:.6f}</li>"
                resume_msg += "</ul>"
                resume_msg += f"<p><strong>Total Trades Completed:</strong> {len(self.trade_history)}</p>"
                self.send_notification("🔄 Bot Resumed - Trade Ongoing", resume_msg)
            else:
                print("✅ State loaded - no active positions")
            
        except Exception as e:
            print(f"⚠️ Failed to load state: {e} - starting fresh")

    async def get_balance(self, currency='USDT'):
        """Fetches current balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance['total'].get(currency, 0)
        except Exception as e:
            print(f"❌ Balance fetch failed: {e}")
            return 0

    async def analyze_volatility(self):
        """
        Finds top 20 mid-tier coins with high volatility and potential upside.
        Filters for coins at relative lows with strong volume.
        """
        print("🔍 Scanning for volatile mid-tier coins...")
        try:
            tickers = await self.exchange.fetch_tickers()
            
            candidates = []
            for symbol, data in tickers.items():
                # Filter: USDT pairs only, must have volume and price data
                if '/USDT' not in symbol or not all([
                    data.get('percentage'),
                    data.get('quoteVolume'),
                    data.get('high'),
                    data.get('low'),
                    data.get('last')
                ]):
                    continue
                
                price = data['last']
                high_24h = data['high']
                low_24h = data['low']
                volume_24h = data['quoteVolume']
                change_24h = data['percentage']
                
                # Skip stablecoins and very low volume coins
                if volume_24h < 100000:  # Minimum $100k daily volume
                    continue
                
                # Calculate where price is in 24h range (0 = at low, 1 = at high)
                if high_24h == low_24h:
                    position_in_range = 0.5
                else:
                    position_in_range = (price - low_24h) / (high_24h - low_24h)
                
                # Calculate volatility score
                volatility = abs(change_24h)
                
                # We want: high volatility, currently near lows, decent volume
                # Score favors coins at lower end of their range
                score = (volatility * 0.4) + ((1 - position_in_range) * 0.4) + (min(volume_24h / 1000000, 10) * 0.2)
                
                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'change_24h': change_24h,
                    'volatility': volatility,
                    'position_in_range': position_in_range,
                    'volume': volume_24h,
                    'score': score,
                    'high_24h': high_24h,
                    'low_24h': low_24h
                })
            
            # Sort by score and take top 20
            candidates.sort(key=lambda x: x['score'], reverse=True)
            self.watchlist = candidates[:self.num_coins]
            
            # Send detailed email report
            report = self.generate_watchlist_report(self.watchlist)
            self.send_notification("🔍 Watchlist Updated - Top Volatile Coins", report)
            
            self.last_scan_time = datetime.now()
            self.save_state()  # Save state after updating watchlist
            print(f"✅ Watchlist updated with {len(self.watchlist)} coins")
            
        except Exception as e:
            print(f"❌ Volatility scan failed: {e}")

    def generate_watchlist_report(self, watchlist):
        """Generates HTML report of watchlist"""
        html = "<h2>🎯 Top 20 Volatile Coins Analysis</h2>"
        html += f"<p><strong>Scan Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        html += "<table border='1' cellpadding='5' style='border-collapse: collapse;'>"
        html += "<tr style='background-color: #f2f2f2;'>"
        html += "<th>Rank</th><th>Symbol</th><th>Price</th><th>24h Change</th><th>Position in Range</th><th>Volume (24h)</th><th>Score</th>"
        html += "</tr>"
        
        for idx, coin in enumerate(watchlist, 1):
            position_pct = coin['position_in_range'] * 100
            position_color = 'green' if position_pct < 30 else 'orange' if position_pct < 60 else 'red'
            
            html += f"<tr>"
            html += f"<td>{idx}</td>"
            html += f"<td><strong>{coin['symbol']}</strong></td>"
            html += f"<td>${coin['price']:.6f}</td>"
            html += f"<td style='color: {'green' if coin['change_24h'] > 0 else 'red'};'>{coin['change_24h']:.2f}%</td>"
            html += f"<td style='color: {position_color};'>{position_pct:.1f}%</td>"
            html += f"<td>${coin['volume']:,.0f}</td>"
            html += f"<td>{coin['score']:.2f}</td>"
            html += f"</tr>"
        
        html += "</table>"
        html += "<br><p><em>Note: Lower 'Position in Range' means coin is closer to 24h low (better entry potential)</em></p>"
        return html

    async def check_min_order_size(self, symbol, current_balance, price):
        """Checks if balance meets minimum order requirements"""
        try:
            market = self.exchange.market(symbol)
            
            min_cost = market['limits']['cost']['min'] or 0
            min_amount = market['limits']['amount']['min'] or 0
            calculated_min_cost = min_amount * price if min_amount else 0
            
            required_usdt = max(min_cost, calculated_min_cost) * 1.02  # 2% buffer
            
            if current_balance < required_usdt:
                print(f"⚠️ Skipping {symbol}: Need ${required_usdt:.2f}, have ${current_balance:.2f}")
                return False
            
            return True
        except Exception as e:
            print(f"⚠️ Could not check limits for {symbol}: {e}")
            return False

    async def get_rsi(self, symbol, period=14):
        """Calculates RSI indicator"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=period + 1)
            closes = [candle[4] for candle in ohlcv]
            
            gains = []
            losses = []
            
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = statistics.mean(gains) if gains else 0
            avg_loss = statistics.mean(losses) if losses else 0
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            print(f"⚠️ RSI calculation failed for {symbol}: {e}")
            return 50  # Neutral RSI

    async def check_entry_conditions(self, coin_data):
        """
        Advanced entry conditions:
        - RSI oversold (< 35)
        - Price near 24h low
        - Positive volume
        """
        symbol = coin_data['symbol']
        
        try:
            # Get RSI
            rsi = await self.get_rsi(symbol)
            
            # Conditions
            rsi_oversold = rsi < 35
            near_low = coin_data['position_in_range'] < 0.35  # In bottom 35% of range
            high_volume = coin_data['volume'] > 500000  # Above $500k volume
            
            print(f"📊 {symbol} - RSI: {rsi:.1f}, Position: {coin_data['position_in_range']*100:.1f}%, Volume: ${coin_data['volume']:,.0f}")
            
            if rsi_oversold and near_low and high_volume:
                print(f"✅ Entry signal for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Entry check failed for {symbol}: {e}")
            return False

    async def open_position(self, coin_data):
        """Opens a new position"""
        symbol = coin_data['symbol']
        
        try:
            balance = await self.get_balance()
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Check minimum order size
            if not await self.check_min_order_size(symbol, balance, price):
                return
            
            print(f"🚀 Opening position: {symbol}")
            
            # Calculate quantity
            qty = self.exchange.amount_to_precision(symbol, (balance * self.risk_per_trade) / price)
            
            # Execute market buy
            order = await self.exchange.create_market_buy_order(symbol, qty)
            entry_price = order['average'] if order.get('average') else price
            
            # Store position
            pos_id = f"{symbol}_{int(datetime.now().timestamp())}"
            self.positions[pos_id] = {
                'symbol': symbol,
                'quantity': float(qty),
                'entry_price': entry_price,
                'take_profit': entry_price * (1 + self.profit_target),
                'stop_loss': entry_price * (1 - self.stop_loss),
                'entry_time': datetime.now(),
                'rsi_entry': await self.get_rsi(symbol),
                'position_in_range': coin_data['position_in_range']
            }
            
            # Send detailed entry email
            entry_report = self.generate_entry_report(self.positions[pos_id], coin_data)
            self.send_notification(f"🎯 TRADE OPENED: {symbol}", entry_report)
            
            self.save_state()  # Save state after opening position
            print(f"✅ Position opened: {symbol} at ${entry_price}")
            
        except Exception as e:
            print(f"❌ Failed to open position for {symbol}: {e}")

    def generate_entry_report(self, position, coin_data):
        """Generates detailed entry report"""
        html = f"<h2>🎯 Trade Opened: {position['symbol']}</h2>"
        html += f"<p><strong>Time:</strong> {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>"
        html += "<h3>Entry Details:</h3>"
        html += "<ul>"
        html += f"<li><strong>Entry Price:</strong> ${position['entry_price']:.6f}</li>"
        html += f"<li><strong>Quantity:</strong> {position['quantity']:.4f}</li>"
        html += f"<li><strong>Take Profit:</strong> ${position['take_profit']:.6f} (+3%)</li>"
        html += f"<li><strong>Stop Loss:</strong> ${position['stop_loss']:.6f} (-3%)</li>"
        html += "</ul>"
        html += "<h3>Technical Analysis:</h3>"
        html += "<ul>"
        html += f"<li><strong>RSI:</strong> {position['rsi_entry']:.2f}</li>"
        html += f"<li><strong>Position in 24h Range:</strong> {position['position_in_range']*100:.1f}%</li>"
        html += f"<li><strong>24h Change:</strong> {coin_data['change_24h']:.2f}%</li>"
        html += f"<li><strong>24h Volume:</strong> ${coin_data['volume']:,.0f}</li>"
        html += f"<li><strong>Volatility Score:</strong> {coin_data['score']:.2f}</li>"
        html += "</ul>"
        return html

    async def close_position(self, pos_id, current_price, reason):
        """Closes a position and sends detailed report"""
        position = self.positions[pos_id]
        symbol = position['symbol']
        
        try:
            print(f"🔄 Closing position: {symbol} ({reason})")
            
            # Execute market sell
            order = await self.exchange.create_market_sell_order(symbol, position['quantity'])
            exit_price = order['average'] if order.get('average') else current_price
            
            # Calculate profit/loss
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            pnl_usd = (exit_price - position['entry_price']) * position['quantity']
            
            # Get current balance
            current_balance = await self.get_balance()
            
            # Store trade history
            trade_record = {
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'reason': reason,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'duration': datetime.now() - position['entry_time'],
                'balance_after': current_balance
            }
            self.trade_history.append(trade_record)
            
            # Generate and send exit report
            exit_report = self.generate_exit_report(trade_record, position)
            emoji = "🎉" if pnl_pct > 0 else "⚠️"
            self.send_notification(f"{emoji} TRADE CLOSED: {symbol} ({reason})", exit_report)
            
            # Remove position
            del self.positions[pos_id]
            
            self.save_state()  # Save state after closing position
            print(f"✅ Position closed: {symbol} | P&L: {pnl_pct:.2f}% (${pnl_usd:.2f})")
            
        except Exception as e:
            print(f"❌ Failed to close position {symbol}: {e}")

    def generate_exit_report(self, trade, position):
        """Generates detailed exit report"""
        pnl_color = 'green' if trade['pnl_pct'] > 0 else 'red'
        
        html = f"<h2>{'🎉' if trade['pnl_pct'] > 0 else '⚠️'} Trade Closed: {trade['symbol']}</h2>"
        html += f"<p><strong>Exit Reason:</strong> {trade['reason']}</p>"
        html += f"<p><strong>Exit Time:</strong> {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        html += "<h3>Trade Summary:</h3>"
        html += "<table border='1' cellpadding='5' style='border-collapse: collapse;'>"
        html += f"<tr><td><strong>Entry Price</strong></td><td>${trade['entry_price']:.6f}</td></tr>"
        html += f"<tr><td><strong>Exit Price</strong></td><td>${trade['exit_price']:.6f}</td></tr>"
        html += f"<tr><td><strong>Quantity</strong></td><td>{trade['quantity']:.4f}</td></tr>"
        html += f"<tr style='background-color: {'#d4edda' if trade['pnl_pct'] > 0 else '#f8d7da'};'>"
        html += f"<td><strong>Profit/Loss</strong></td><td style='color: {pnl_color};'><strong>{trade['pnl_pct']:.2f}% (${trade['pnl_usd']:.2f})</strong></td></tr>"
        html += f"<tr><td><strong>Duration</strong></td><td>{str(trade['duration']).split('.')[0]}</td></tr>"
        html += f"<tr><td><strong>Current Balance</strong></td><td>${trade['balance_after']:.2f}</td></tr>"
        html += "</table>"
        
        # Add performance stats if we have trade history
        if len(self.trade_history) > 1:
            html += self.generate_performance_stats()
        
        return html

    def generate_performance_stats(self):
        """Generates overall performance statistics"""
        wins = [t for t in self.trade_history if t['pnl_pct'] > 0]
        losses = [t for t in self.trade_history if t['pnl_pct'] <= 0]
        
        total_trades = len(self.trade_history)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = statistics.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = statistics.mean([t['pnl_pct'] for t in losses]) if losses else 0
        total_pnl = sum([t['pnl_usd'] for t in self.trade_history])
        
        html = "<h3>📊 Performance Statistics:</h3>"
        html += "<ul>"
        html += f"<li><strong>Total Trades:</strong> {total_trades}</li>"
        html += f"<li><strong>Wins:</strong> {len(wins)} | <strong>Losses:</strong> {len(losses)}</li>"
        html += f"<li><strong>Win Rate:</strong> {win_rate:.1f}%</li>"
        html += f"<li><strong>Average Win:</strong> {avg_win:.2f}%</li>"
        html += f"<li><strong>Average Loss:</strong> {avg_loss:.2f}%</li>"
        html += f"<li><strong>Total P&L:</strong> ${total_pnl:.2f}</li>"
        html += "</ul>"
        
        return html

    async def monitor_positions(self):
        """Monitors open positions for exit conditions"""
        for pos_id, pos in list(self.positions.items()):
            try:
                ticker = await self.exchange.fetch_ticker(pos['symbol'])
                current_price = ticker['last']
                current_pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                
                print(f"📈 {pos['symbol']}: ${current_price:.6f} | P&L: {current_pnl:.2f}%")
                
                # Check take profit
                if current_price >= pos['take_profit']:
                    await self.close_position(pos_id, current_price, "Take Profit (3%)")
                
                # Check stop loss
                elif current_price <= pos['stop_loss']:
                    await self.close_position(pos_id, current_price, "Stop Loss (3%)")
                    
            except Exception as e:
                print(f"⚠️ Error monitoring {pos['symbol']}: {e}")

    async def start(self):
        """Main bot loop"""
        self.running = True
        await self.exchange.load_markets()
        
        # Only set initial balance if not resuming
        if not self.initial_balance:
            self.initial_balance = await self.get_balance()
            self.save_state()
        
        current_balance = await self.get_balance()
        print(f"🤖 Bot Started | Initial: ${self.initial_balance:.2f} | Current: ${current_balance:.2f}")
        
        # Only send startup email if no positions (fresh start)
        if not self.positions:
            startup_msg = f"<h2>🤖 Volatility Bot Started</h2><p><strong>Starting Balance:</strong> ${self.initial_balance:.2f}</p>"
            startup_msg += f"<p><strong>Strategy:</strong> 3% Take Profit / 3% Stop Loss</p>"
            startup_msg += f"<p><strong>Target:</strong> Top 20 volatile mid-tier coins</p>"
            if len(self.trade_history) > 0:
                startup_msg += f"<p><strong>Previous Trades:</strong> {len(self.trade_history)}</p>"
            self.send_notification("🤖 Bot Online", startup_msg)
        
        while self.running:
            try:
                # 1. Update watchlist every 2 hours
                if datetime.now() > self.last_scan_time + timedelta(hours=self.scan_interval):
                    await self.analyze_volatility()
                
                # 2. Monitor existing positions
                if self.positions:
                    await self.monitor_positions()
                
                # 3. Look for new entries if no position open
                else:
                    for coin in self.watchlist:
                        balance = await self.get_balance()
                        if balance < 5:  # Minimum balance check
                            print("❌ Insufficient balance")
                            break
                        
                        if await self.check_entry_conditions(coin):
                            await self.open_position(coin)
                            break  # Only one position at a time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"⚠️ Loop error: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Gracefully stops the bot"""
        self.running = False
        print("🛑 Bot stopping...")
        
        # Save state before stopping
        self.save_state()
        
        # Only close positions if explicitly requested (not on normal Render restarts)
        # Positions will be preserved and resumed on restart
        print(f"💾 State saved with {len(self.positions)} active position(s)")
        
        await self.exchange.close()

    async def health_check_server(self):
        """Simple HTTP server for Render health checks"""
        from aiohttp import web
        
        async def health(request):
            status = {
                'status': 'running',
                'active_positions': len(self.positions),
                'total_trades': len(self.trade_history),
                'balance': await self.get_balance(),
                'uptime': str(datetime.now() - self.start_time) if hasattr(self, 'start_time') else 'N/A'
            }
            return web.json_response(status)
        
        app = web.Application()
        app.router.add_get('/health', health)
        app.router.add_get('/', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.environ.get('PORT', 10000)))
        await site.start()
        print(f"🌐 Health check server running on port {os.environ.get('PORT', 10000)}")

# --- EXECUTION ---
if __name__ == "__main__":
    bot = OKXVolatilityBot(
        api_key="50d33da8-d3b5-4803-963c-ef7c19c6cc2a", 
        api_secret="3F8825E8FBEB7C80F56FF32448919ADA", 
        passphrase="Alphaxide@2003", 
        resend_key="re_1URbBz6v_8p6JsriYBLGoxkF2Srq5ZBZ2", 
        target_email="kalphaxide@gmail.com", 
        testnet=False  # Set to True for testing
    )
    
    bot.start_time = datetime.now()
    
    async def run_bot():
        # Start health check server for Render
        await bot.health_check_server()
        # Start trading bot
        await bot.start()
    
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        asyncio.run(bot.stop())
