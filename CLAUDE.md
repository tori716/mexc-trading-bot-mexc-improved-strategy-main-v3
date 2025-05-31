# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Development Commands

### Running the Bot
```bash
# Install dependencies
pip install -r requirements.txt

# Run in test mode (default - uses virtual trading)
python main.py

# Run with specific Python version if needed
python3 main.py
```

### Environment Setup
```bash
# Copy and configure environment variables
cp env_template.sh .env
# Edit .env with your API keys and configuration

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Testing
```bash
# Run tests (if available)
python -m pytest

# Run specific test files
python -m pytest -v
```

## 🏗️ Architecture Overview

This is a MEXC cryptocurrency trading bot with improved strategy implementation (v3) that supports both paper trading (test mode) and live trading.

### Core Components

**main.py** - Entry point that orchestrates the entire bot execution flow
- Initializes logging, configuration, API connections, and trading strategy
- Supports both test mode and live trading mode

**config.py** - Centralized configuration management
- Loads environment variables from .env file with extensive validation
- Manages 80+ configuration parameters including trading strategy parameters, risk management settings, and technical indicators
- Includes coin-specific adjustments and optimized trading hours

**mexc_api.py** - MEXC exchange API interface
- Handles REST API communication with MEXC exchange
- Manages authentication, market data retrieval, and order execution
- Implements caching and concurrent API call management

**strategy.py** - Core trading strategy implementation
- Multi-layered trading strategy with technical indicators (Bollinger Bands, RSI, MACD, EMA)
- Advanced risk management with progressive stop-loss, trailing stops, and time-based exits
- Market condition adaptive thresholds and coin-specific parameter adjustments
- Supports up to 25 monitoring symbols with tiered prioritization

**utils.py** - Utility functions and Discord notifications
- Discord webhook integration for trade notifications and daily reports
- Various helper functions for data processing and logging

**reports.py** - Daily report generation system
- Generates comprehensive daily trading reports
- Calculates performance metrics and statistics

### Key Features

- **Multi-mode Operation**: Test mode (paper trading) and live trading mode
- **Advanced Risk Management**: Progressive stop-loss, trailing stops, position sizing
- **Market Adaptive**: Time-based parameter adjustments and market condition detection
- **Comprehensive Monitoring**: Up to 25 cryptocurrency pairs with tiered prioritization
- **Real-time Notifications**: Discord integration for trade alerts and daily reports
- **Performance Optimization**: API caching, concurrent calls, and efficient data management

### Configuration Structure

The bot uses a tiered symbol monitoring system:
- **Tier 1**: High-priority coins (AVAX, LINK, NEAR, FTM, ATOM, DOT, MATIC, UNI, AAVE, DOGE)
- **Tier 2**: Medium-priority coins (ADA, ALGO, APE, ARB, EGLD, FIL, GRT, ICP, LTC, SAND)  
- **Tier 3**: Lower-priority coins (SHIB, VET, MANA, GALA, ONE)

Trading strategy adapts based on:
- Market conditions (bull/bear/sideways)
- Time of day (optimized trading hours: 15:00-18:00 JST, 22:00-24:00 JST)
- Individual coin characteristics
- Recent performance history

### Environment Variables

Critical configuration is managed through environment variables:
- **MEXC_API_KEY/MEXC_SECRET_KEY**: Required for live trading
- **DISCORD_WEBHOOK_URL**: Required for notifications
- **TEST_MODE**: Controls paper trading vs live trading
- **80+ trading parameters**: Technical indicators, risk management, thresholds

See `env_template.sh` for complete configuration options with descriptions.

### Data Flow

1. Bot starts and loads configuration from environment variables
2. Initializes MEXC API connection and Discord notifier
3. Creates trading strategy instance with configured parameters
4. Enters main trading loop:
   - Fetches market data for monitored symbols
   - Applies technical analysis and market condition detection
   - Executes entry/exit decisions based on strategy rules
   - Manages active positions with risk controls
   - Sends notifications for significant events
5. Generates daily reports and performance summaries

### Security Considerations

- API keys and secrets must be stored in `.env` file (not committed to git)
- Discord webhook URLs are considered sensitive
- Test mode provides safe environment for strategy validation
- Progressive position sizing limits risk exposure