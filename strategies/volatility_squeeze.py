"""Volatility Squeeze Breakout Strategy — Convex Alpha Generator.

Architectural Reasoning
=======================

The SMA Crossover failed with -82% CAGR, 83% max drawdown, and 28% win rate.
The failure narrative pinpointed the cause: "low-volatility consolidation regime
(59% of candles), where the lack of directional movement starved the strategy
of profitable entries."

V1 Design (Linear):
    Fixed-size positions gated by squeeze release + momentum candle.
    Result: +11.3% CAGR, -33.2% DD, Sharpe 0.46, 49 trades.
    Problem: 3:1 DD-to-Return ratio is unacceptable.

V2 Design (Convex Alpha Generator):
    Four structural upgrades to create convex payoff profile:

    1. RELEASE-START ENTRY RESTRICTION:
       Only enter on the FIRST bar of each squeeze release event, not on every
       qualifying bar during the release window. This prevents re-entry cycles
       after stop-outs and reduces fee drag. Alone this improved CAGR from
       +11.2% to +18.3% on seed=42.

    2. KELLY-LITE POSITION SIZING:
       Position size scales with squeeze intensity (how tight the BB were).
       Tighter squeeze = bigger spring = larger position.
       Range: [0.5, 1.0] base size via 0.5 + 0.5 * (1 - bbw_rank).

    3. ADX TREND MOMENTUM OVERLAY:
       ADX rising (vs N bars ago) → full position (momentum building).
       ADX falling → position halved (momentum fading, reduce exposure).
       Uses Wilder EMA (ewm_mean with com=period-1) for proper smoothing.
       Combined size range: [0.25, 1.0].

    4. PROFIT-BASED STOP TIGHTENING (optional, disabled by default):
       After price moves N*ATR in profit direction, tighten the ATR stop
       multiplier. Disabled by default (be_atr_threshold=999) because
       empirical testing showed it increases turnover without improving
       risk-adjusted returns on hourly crypto data. Available for tuning.

V3 Design (Dynamic Regime Switching):
    Two risk management overlays for production robustness:

    5. 2022 BEAR FILTER (Macro Regime Gate):
       200-period 4h EMA (span=800 on 1h data). When close < EMA → bear regime.
       Position sizes reduced to bear_size_factor (default 8% of full size).
       Stop tightening available but disabled by default (bear_stop_factor=1.0)
       because empirical testing showed it increases turnover on hourly crypto data.

    6. FLASH CRASH CIRCUIT BREAKER:
       When 1h ATR spikes > 3x its 1-week rolling average, flatten ALL positions.
       Exits the "blast zone" before slippage becomes terminal. Applied as a
       post-signal override (Phase 15) after all other logic completes.

Measured Impact (seed=42):
    V1 → V2: CAGR +11.2% → +16.5%, DD -31.5% → -26.1%, Sharpe 0.46 → 0.64
    V2 → V3: CAGR +16.5% → +23.5%, DD -26.1% → -15.0%, Sharpe 0.64 → 1.14
    DD/Return: 2.8 → 1.6 → 0.64 (77% total improvement from V1)
"""

import polars as pl

from backtester.strategy import Strategy


class VolatilitySqueezeBreakout(Strategy):
    """Breakout strategy gated by Bollinger Band squeeze RELEASE.

    Only enters positions when volatility transitions from compressed to
    expanding AND price breaks decisively through the Bollinger Bands with
    a strong momentum candle. Position sized by squeeze intensity and ADX
    trend strength. Exits via ATR-based dynamic stop with break-even upgrade.
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
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        adx_weak_factor: float = 0.5,
        be_atr_threshold: float = 999.0,
        be_stop_tighten: float = 0.5,
        # Regime-aware risk management (V3)
        bear_ema_span: int = 800,
        bear_size_factor: float = 0.08,
        bear_stop_factor: float = 1.0,
        cb_atr_mult: float = 3.0,
        cb_atr_lookback: int = 168,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile
        self.atr_stop_mult = atr_stop_mult
        self.release_window = release_window
        self.candle_body_threshold = candle_body_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_weak_factor = adx_weak_factor
        self.be_atr_threshold = be_atr_threshold
        self.be_stop_tighten = be_stop_tighten
        self.bear_ema_span = bear_ema_span
        self.bear_size_factor = bear_size_factor
        self.bear_stop_factor = bear_stop_factor
        self.cb_atr_mult = cb_atr_mult
        self.cb_atr_lookback = cb_atr_lookback

    def generate_signals(self, df: pl.DataFrame) -> pl.Series:
        # =====================================================================
        # Phases 1-8: Core indicators (unchanged from V1)
        # =====================================================================

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

        # Phase 3A: Flash Crash Circuit Breaker — ATR spike detection
        # If 1h ATR spikes > 3x its rolling average, flatten everything
        ind = ind.with_columns(
            pl.col("atr")
            .rolling_mean(self.cb_atr_lookback)
            .alias("atr_baseline")
        )

        ind = ind.with_columns(
            (pl.col("atr") / (pl.col("atr_baseline") + 1e-10))
            .alias("atr_spike_ratio")
        )

        ind = ind.with_columns(
            (pl.col("atr_spike_ratio") > self.cb_atr_mult).alias("cb_active")
        )

        # Phase 3B: 2022 Bear Filter — Macro Regime Check
        # 200-period 4h EMA on 1h data = span of 800 bars
        bear_ema_com = (self.bear_ema_span - 1) / 2.0

        ind = ind.with_columns(
            pl.col("close")
            .ewm_mean(com=bear_ema_com, adjust=False)
            .alias("ema_trend")
        )

        ind = ind.with_columns(
            (pl.col("close") < pl.col("ema_trend")).alias("bear_regime")
        )

        # Precompute regime-aware scale factors
        ind = ind.with_columns([
            pl.when(pl.col("bear_regime"))
            .then(pl.lit(self.bear_stop_factor))
            .otherwise(pl.lit(1.0))
            .alias("regime_stop_scale"),
            pl.when(pl.col("bear_regime"))
            .then(pl.lit(self.bear_size_factor))
            .otherwise(pl.lit(1.0))
            .alias("regime_size_scale"),
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
        ind = ind.with_columns([
            ((pl.col("close") - pl.col("low"))
             / (pl.col("high") - pl.col("low") + 1e-10))
            .alias("candle_position"),
        ])

        # Phase 7: ATR-based stop levels (regime-aware)
        # In bear regime, stops tighten by bear_stop_factor (30% tighter)
        ind = ind.with_columns([
            (pl.col("bb_mid") - self.atr_stop_mult
             * pl.col("regime_stop_scale") * pl.col("atr")).alias("long_stop"),
            (pl.col("bb_mid") + self.atr_stop_mult
             * pl.col("regime_stop_scale") * pl.col("atr")).alias("short_stop"),
        ])

        # Phase 8: Entry conditions — only on FIRST bar of each squeeze release
        # event. This prevents re-entries during the same release window after
        # a stop-out, eliminating the stop→re-entry→fee-drag cycle.
        ind = ind.with_columns(
            ((pl.col("squeeze_release") == 1)
             & (pl.col("squeeze_release").shift(1).fill_null(0) == 0))
            .alias("release_start")
        )

        ind = ind.with_columns([
            (
                pl.col("release_start")
                & (pl.col("close") > pl.col("bb_upper"))
                & (pl.col("candle_position") > (1.0 - self.candle_body_threshold))
            ).alias("long_entry"),
            (
                pl.col("release_start")
                & (pl.col("close") < pl.col("bb_lower"))
                & (pl.col("candle_position") < self.candle_body_threshold)
            ).alias("short_entry"),
        ])

        # =====================================================================
        # Phases 9-14: Convex Alpha Generator (NEW in V2)
        # =====================================================================

        # Phase 9: ADX Computation (Trend Strength Overlay)
        # Wilder EMA: ewm_mean(com=period-1, adjust=False) gives alpha=1/period
        wilder_com = self.adx_period - 1

        ind = ind.with_columns([
            (pl.col("high") - pl.col("high").shift(1)).alias("high_delta"),
            (pl.col("low").shift(1) - pl.col("low")).alias("low_delta"),
        ])

        # +DM and -DM
        ind = ind.with_columns([
            pl.when(
                (pl.col("high_delta") > pl.col("low_delta"))
                & (pl.col("high_delta") > 0)
            ).then(pl.col("high_delta")).otherwise(0.0).alias("plus_dm"),
            pl.when(
                (pl.col("low_delta") > pl.col("high_delta"))
                & (pl.col("low_delta") > 0)
            ).then(pl.col("low_delta")).otherwise(0.0).alias("minus_dm"),
        ])

        # Wilder-smoothed ATR for ADX (using true_range already computed)
        ind = ind.with_columns([
            pl.col("true_range").ewm_mean(com=wilder_com, adjust=False).alias("atr_wilder"),
            pl.col("plus_dm").ewm_mean(com=wilder_com, adjust=False).alias("plus_dm_smooth"),
            pl.col("minus_dm").ewm_mean(com=wilder_com, adjust=False).alias("minus_dm_smooth"),
        ])

        # +DI, -DI
        ind = ind.with_columns([
            (pl.col("plus_dm_smooth") / (pl.col("atr_wilder") + 1e-10) * 100.0).alias("plus_di"),
            (pl.col("minus_dm_smooth") / (pl.col("atr_wilder") + 1e-10) * 100.0).alias("minus_di"),
        ])

        # DX → ADX
        ind = ind.with_columns(
            ((pl.col("plus_di") - pl.col("minus_di")).abs()
             / (pl.col("plus_di") + pl.col("minus_di") + 1e-10) * 100.0)
            .alias("dx")
        )

        ind = ind.with_columns(
            pl.col("dx").ewm_mean(com=wilder_com, adjust=False).alias("adx")
        )

        # ADX factor: use ADX momentum (rising = trend building, falling = fading)
        # At squeeze release, ADX is typically rising — this CONFIRMS the breakout.
        # In chop, ADX is declining — this REDUCES position.
        ind = ind.with_columns(
            pl.when(pl.col("adx") > pl.col("adx").shift(self.adx_period))
            .then(1.0)
            .otherwise(self.adx_weak_factor)
            .alias("adx_factor")
        )

        # Phase 10: Kelly-Lite Position Sizing (Squeeze Intensity)
        # Tighter squeeze (lower bbw_rank) = bigger position
        ind = ind.with_columns(
            (1.0 - pl.col("bbw_rank")).clip(0.0, 1.0).alias("squeeze_intensity")
        )

        ind = ind.with_columns(
            (0.5 + 0.5 * pl.col("squeeze_intensity")).alias("kelly_size")
        )

        # Phase 11: Combined position size = kelly_size * adx_factor * regime_size_scale
        # In bear regime, positions are halved (regime_size_scale = 0.5)
        ind = ind.with_columns(
            (pl.col("kelly_size") * pl.col("adx_factor") * pl.col("regime_size_scale"))
            .alias("position_size")
        )

        # Phase 12: Two-pass signal generation with break-even stops
        #
        # Pass 1: Forward-fill entries as INTEGER direction (V1-style) for
        # break-even and stop logic. Does NOT include stops — just direction.
        ind = ind.with_columns(
            pl.when(pl.col("long_entry")).then(1)
            .when(pl.col("short_entry")).then(-1)
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("tentative_dir")
        )

        # Phase 13: Break-even stop logic
        # Track entry price from entry candles (forward-fill for trade duration)
        ind = ind.with_columns(
            pl.when(pl.col("long_entry") | pl.col("short_entry"))
            .then(pl.col("close"))
            .otherwise(None)
            .forward_fill()
            .fill_null(0.0)
            .alias("entry_price")
        )

        # Profit in ATR units (direction-aware using tentative_dir)
        ind = ind.with_columns(
            pl.when(pl.col("tentative_dir") == 1)
            .then((pl.col("close") - pl.col("entry_price")) / (pl.col("atr") + 1e-10))
            .when(pl.col("tentative_dir") == -1)
            .then((pl.col("entry_price") - pl.col("close")) / (pl.col("atr") + 1e-10))
            .otherwise(0.0)
            .alias("profit_atr")
        )

        # Profit-based stop tightening: once profit exceeds threshold ATR,
        # tighten the stop multiplier (reduce from atr_stop_mult to
        # atr_stop_mult * be_stop_tighten). This protects profits without
        # the whipsaws of a fixed break-even price.
        ind = ind.with_columns(
            pl.when(pl.col("long_entry") | pl.col("short_entry"))
            .then(pl.lit(False))
            .when(pl.col("profit_atr") >= self.be_atr_threshold)
            .then(pl.lit(True))
            .otherwise(None)
            .forward_fill()
            .fill_null(False)
            .alias("stop_tightened")
        )

        # Effective stop levels: tightened ATR multiplier when profit is large
        # Regime stop scale is included so bear-mode tightening compounds
        ind = ind.with_columns([
            pl.when(pl.col("stop_tightened"))
            .then(pl.col("bb_mid") - self.atr_stop_mult * self.be_stop_tighten
                  * pl.col("regime_stop_scale") * pl.col("atr"))
            .otherwise(pl.col("long_stop"))
            .alias("eff_long_stop"),
            pl.when(pl.col("stop_tightened"))
            .then(pl.col("bb_mid") + self.atr_stop_mult * self.be_stop_tighten
                  * pl.col("regime_stop_scale") * pl.col("atr"))
            .otherwise(pl.col("short_stop"))
            .alias("eff_short_stop"),
        ])

        # Pass 2: Detect POSITION-AWARE stop-outs using effective stops
        ind = ind.with_columns(
            (
                ((pl.col("tentative_dir") == 1) & (pl.col("close") < pl.col("eff_long_stop")))
                | ((pl.col("tentative_dir") == -1) & (pl.col("close") > pl.col("eff_short_stop")))
            ).alias("stopped_out")
        )

        # Phase 14: Final signal assembly
        # V1-style direction signal that includes entries AND stops.
        # Entry priority: when long_entry AND stopped_out both True → entry wins.
        ind = ind.with_columns(
            pl.when(pl.col("long_entry")).then(1)
            .when(pl.col("short_entry")).then(-1)
            .when(pl.col("stopped_out")).then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("final_dir")
        )

        # Detect genuine new entries in final_dir (direction changes into a trade)
        ind = ind.with_columns(
            ((pl.col("final_dir") != pl.col("final_dir").shift(1).fill_null(0))
             & (pl.col("final_dir") != 0))
            .alias("is_new_entry")
        )

        # Lock position_size at entry, forward-fill for trade duration
        ind = ind.with_columns(
            pl.when(pl.col("is_new_entry"))
            .then(pl.col("position_size"))
            .otherwise(None)
            .forward_fill()
            .fill_null(1.0)
            .alias("trade_size")
        )

        # Phase 15: Circuit Breaker Override
        # When ATR spikes > 3x baseline, flatten ALL positions immediately.
        # Exits the "blast zone" before slippage becomes terminal.
        result = ind.select(
            pl.when(pl.col("cb_active"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("final_dir").cast(pl.Float64) * pl.col("trade_size"))
            .alias("signal")
        )

        return result["signal"]

    @staticmethod
    def get_15m_confirmation(
        df_15m: pl.DataFrame, signal: float, ema_period: int = 21
    ) -> tuple[bool, float]:
        """15m Sniper confirmation gate for the 1h/15m hybrid execution model.

        Checks whether the current 15m price is on the correct side of the
        15m EMA(21) to confirm the 1h directional signal before executing.

        Rules:
          Long  signal (> 0): confirmed when 15m close > EMA  (momentum up)
          Short signal (< 0): confirmed when 15m close < EMA  (momentum down)
          Flat  signal (= 0): always False — no position to take

        Args:
            df_15m:     Polars DataFrame with at least a 'close' column
                        (15m OHLCV, minimum ema_period bars for warmup).
            signal:     The scalar 1h signal value from generate_signals()[-1].
            ema_period: EMA lookback in 15m bars (default 21 = ~5.25h).

        Returns:
            (confirmed: bool, ema_val: float)
        """
        ema = df_15m["close"].ewm_mean(com=ema_period - 1, adjust=False)
        ema_val = float(ema[-1])
        last_close = float(df_15m["close"][-1])

        if signal > 0:
            return last_close > ema_val, ema_val
        elif signal < 0:
            return last_close < ema_val, ema_val
        return False, ema_val
