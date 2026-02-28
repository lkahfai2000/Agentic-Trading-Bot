"""Volatility Squeeze Breakout Strategy.

Architectural Reasoning
=======================

The SMA Crossover failed with -82% CAGR, 83% max drawdown, and 28% win rate.
The failure narrative pinpointed the cause: "low-volatility consolidation regime
(59% of candles), where the lack of directional movement starved the strategy
of profitable entries."

Root Cause Analysis (3 structural flaws):
    1. ALWAYS IN MARKET: SMA crossover has no "flat" state. During the 59%
       consolidation regime, the MAs oscillated around each other producing
       ~100+ whipsaw trades. Each trade lost ~20bps in friction, compounding
       into a 33.6% fee drag floor (168 trades x 20bps).
    2. LAGGING ENTRY: Moving average crossovers are a lagging indicator — by
       the time the fast MA crosses the slow MA, the move is already underway
       and often about to reverse in choppy conditions.
    3. NO INDEPENDENT EXIT: The only way to exit long is to go short (and vice
       versa). Every exit also opens a new position, often into the very chop
       that caused the exit.

This Strategy's Counter-Design:

    1. SQUEEZE RELEASE DETECTION (not just "in squeeze"):
       Instead of entering when volatility is low, we enter when volatility
       TRANSITIONS from compressed to expanding — the "spring uncoils." This
       is the Bollinger Band Width (BBW) transitioning from its bottom decile
       to expansion. This is a rare, high-conviction structural event.

    2. BREAKOUT CONFIRMATION (momentum candle):
       After squeeze release, only enter if:
       - Long: close > upper BB AND candle closes in upper 30% of its range
       - Short: close < lower BB AND candle closes in lower 30% of its range
       This filters out weak/indecisive breakouts.

    3. POSITION-AWARE ATR EXIT (two-pass vectorized):
       Exit to FLAT when price retraces N * ATR from the Bollinger midline.
       Critically, exits are DIRECTION-AWARE: long stops only fire when long,
       short stops only fire when short. Implemented via a two-pass approach:
       Pass 1: Forward-fill entries to establish tentative positions.
       Pass 2: Check stops against tentative position direction, merge with
       entries, forward-fill the combined stream. All vectorized, zero loops.

    4. POSITION PERSISTENCE via forward-fill:
       Between sparse entry/exit events, position is held constant via
       Polars `forward_fill()` — fully vectorized, O(n) complexity.

Expected Impact vs SMA Crossover:
    - Trades: 168 -> ~20-40 (80%+ reduction in fee drag)
    - Win rate: 28% -> 40-55% (only high-conviction breakouts)
    - Max drawdown: dramatically reduced (flat during consolidation)
    - Sharpe: improved (eliminating the chop bleed)
"""

import polars as pl

from backtester.strategy import Strategy


class VolatilitySqueezeBreakout(Strategy):
    """Breakout strategy gated by Bollinger Band squeeze RELEASE.

    Only enters positions when volatility transitions from compressed to
    expanding AND price breaks decisively through the Bollinger Bands with
    a strong momentum candle. Exits to flat via ATR-based dynamic stop.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        squeeze_lookback: int = 240,
        squeeze_pctile: float = 0.10,
        atr_stop_mult: float = 3.5,
        release_window: int = 3,
        candle_body_threshold: float = 0.30,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile
        self.atr_stop_mult = atr_stop_mult
        self.release_window = release_window
        self.candle_body_threshold = candle_body_threshold

    def generate_signals(self, df: pl.DataFrame) -> pl.Series:
        # Phase 1: Core indicators
        ind = df.with_columns([
            pl.col("close").rolling_mean(self.bb_period).alias("bb_mid"),
            pl.col("close").rolling_std(self.bb_period).alias("bb_std_val"),
            (pl.col("high") - pl.col("low")).alias("tr_hl"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_hc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_lc"),
        ])

        # Phase 2: Derived — BB bands, True Range
        ind = ind.with_columns([
            (pl.col("bb_mid") + self.bb_std * pl.col("bb_std_val")).alias("bb_upper"),
            (pl.col("bb_mid") - self.bb_std * pl.col("bb_std_val")).alias("bb_lower"),
            pl.max_horizontal("tr_hl", "tr_hc", "tr_lc").alias("true_range"),
        ])

        # Phase 3: ATR + BBW
        ind = ind.with_columns([
            pl.col("true_range").rolling_mean(self.atr_period).alias("atr"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")).alias("bbw"),
        ])

        # Phase 4: Squeeze detection — BBW in bottom decile of its rolling range
        ind = ind.with_columns([
            pl.col("bbw").rolling_min(self.squeeze_lookback).alias("bbw_min"),
            pl.col("bbw").rolling_max(self.squeeze_lookback).alias("bbw_max"),
        ])

        ind = ind.with_columns(
            ((pl.col("bbw") - pl.col("bbw_min"))
             / (pl.col("bbw_max") - pl.col("bbw_min") + 1e-10))
            .alias("bbw_rank")
        )

        ind = ind.with_columns(
            (pl.col("bbw_rank") < self.squeeze_pctile).cast(pl.Int32).alias("in_squeeze")
        )

        # Phase 5: Squeeze RELEASE — was in squeeze recently, now expanding
        # "Release" = squeeze was active in the last N candles, but is NOT active now
        ind = ind.with_columns([
            pl.col("in_squeeze")
                .rolling_max(self.release_window)
                .fill_null(0)
                .cast(pl.Int32)
                .alias("was_squeezed"),
        ])

        ind = ind.with_columns(
            ((pl.col("was_squeezed") == 1) & (pl.col("in_squeeze") == 0))
            .cast(pl.Int32)
            .alias("squeeze_release")
        )

        # Phase 6: Momentum candle confirmation
        # Long candle = close in upper portion of range
        # Short candle = close in lower portion of range
        ind = ind.with_columns([
            ((pl.col("close") - pl.col("low"))
             / (pl.col("high") - pl.col("low") + 1e-10))
            .alias("candle_position"),
        ])

        # Phase 7: ATR-based stop levels
        ind = ind.with_columns([
            (pl.col("bb_mid") - self.atr_stop_mult * pl.col("atr")).alias("long_stop"),
            (pl.col("bb_mid") + self.atr_stop_mult * pl.col("atr")).alias("short_stop"),
        ])

        # Phase 8: Entry conditions as boolean columns
        ind = ind.with_columns([
            (
                (pl.col("squeeze_release") == 1)
                & (pl.col("close") > pl.col("bb_upper"))
                & (pl.col("candle_position") > (1.0 - self.candle_body_threshold))
            ).alias("long_entry"),
            (
                (pl.col("squeeze_release") == 1)
                & (pl.col("close") < pl.col("bb_lower"))
                & (pl.col("candle_position") < self.candle_body_threshold)
            ).alias("short_entry"),
        ])

        # Phase 9: Two-pass signal generation (position-aware exits)
        #
        # Pass 1: Forward-fill entries only → tentative positions.
        # This tells us "what direction are we in" so exits can be directional.
        ind = ind.with_columns(
            pl.when(pl.col("long_entry")).then(1)
            .when(pl.col("short_entry")).then(-1)
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("tentative_pos")
        )

        # Pass 2: Detect POSITION-AWARE stop-outs.
        # Long stop only fires when tentative position is long.
        # Short stop only fires when tentative position is short.
        ind = ind.with_columns(
            (
                ((pl.col("tentative_pos") == 1) & (pl.col("close") < pl.col("long_stop")))
                | ((pl.col("tentative_pos") == -1) & (pl.col("close") > pl.col("short_stop")))
            ).alias("stopped_out")
        )

        # Merge entries + stops → single event stream → forward fill.
        # Priority: entries override stops (when chain order).
        # After a stop, position goes to 0 and stays flat until next entry.
        result = ind.select(
            pl.when(pl.col("long_entry")).then(1)
            .when(pl.col("short_entry")).then(-1)
            .when(pl.col("stopped_out")).then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("signal")
        )

        return result["signal"]
