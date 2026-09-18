"""quantsoc - shared code for the QuantSoc workshop "Build Your First Quant Trading Strategy".

The package is deliberately small.  Each module maps to one stage of the workshop:

- market     : the controlled synthetic teaching market (generation + loading)
- features   : the ONE feature implementation shared by research, export and the runner
- modeling   : chronological splits, baseline, linear model helpers
- portfolio  : reference allocation rule + constraint checks
- backtest   : the simulation engine used by evaluation AND replay
- artifacts  : export / import of a strategy bundle (code + model + config)
- engine     : the trading engine (reconciliation, order sizing, safety checks)
- broker     : broker adapters (Alpaca paper, replay, mock)
- exercises  : exercise checks, hints and labelled reference solutions
- viz / widgets : plots and interactive controls for the notebook
"""

__version__ = "0.1.0"
SCHEMA_VERSION = 1
