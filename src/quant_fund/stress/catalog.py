"""Historical crisis catalog and one published supervisory scenario.

Equity, volatility, and credit-spread *paths* from vendor indices are not
bundled: those series are not Board of Governors works, and this repository
does not redistribute them. Where a primary official document states a scalar,
that scalar is stored as a factor shock with the quotation's precision and a
qualifier. Where it does not, the entry says so.

Rate and FX *paths* that are H.15 / H.10 releases live in the public-domain
bundle and are applied by the replay layer, not duplicated here.

Nothing in this catalog is a live-trading result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorShock:
    """One cited scenario scalar.

    ``role="primary"`` shocks are applied to a strategy. ``role="context"``
    shocks are reported and are not added on top of the primary shock.
    ``unit`` is one of ``simple_return``, ``level_change``, ``index_level``,
    ``rate_change_pp``, or ``fraction``.
    """

    factor: str
    value: float
    unit: str
    role: str
    qualifier: str
    citation: str
    source_url: str
    licence: str


@dataclass(frozen=True)
class Crisis:
    """One named episode or a clearly labelled hypothetical scenario."""

    crisis_id: str
    name: str
    start: str
    end: str
    historical: bool
    summary: str
    shocks: tuple[FactorShock, ...]
    window_id: str | None
    limitations: tuple[str, ...]


_PD = "U.S. government work (public domain)."
_BIS = "BIS Quarterly Review. Cite the Bank for International Settlements."
_SJES = "Open-access journal article. Cite the authors; a single quoted print, not a redistributed vendor tape."

_FED_HISTORY_1987 = "https://www.federalreservehistory.org/essays/stock-market-crash-of-1987"
_BRADY = (
    "Report of the Presidential Task Force on Market Mechanisms (Brady Commission), January 1988: "
    "on 19 October 1987 the Dow fell 508 points, or 22.6 percent; from the 13 October close to the "
    "19 October close the Dow fell 769 points, or 31 percent."
)
_GREAT_RECESSION = "https://www.federalreservehistory.org/essays/great-recession"
_FLASH = "https://www.sec.gov/files/marketevents-report.pdf"
_AUG24 = "https://www.sec.gov/marketstructure/research/equity_market_volatility.pdf"
_SNB = "https://www.snb.ch/en/publications/communication/press-releases/2015/pre_20150115"
_SJES_URL = "https://link.springer.com/article/10.1186/s41937-025-00137-6"
_BIS_URL = "https://www.bis.org/publ/qtrpdf/r_qt1803a.pdf"
_FOMC_2018 = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180321.htm"
_FEDS_COVID = "https://www.federalreserve.gov/econres/feds/files/2021035pap.pdf"
_FOMC_2022 = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20221214a.htm"
_DFAST = "https://www.federalreserve.gov/publications/files/2025-stress-test-scenarios-20250205.pdf"


def _shock(
    factor: str,
    value: float,
    unit: str,
    role: str,
    qualifier: str,
    citation: str,
    source_url: str,
    licence: str,
) -> FactorShock:
    return FactorShock(factor, value, unit, role, qualifier, citation, source_url, licence)


CRISIS_CATALOG: tuple[Crisis, ...] = (
    Crisis(
        crisis_id="crash_1987",
        name="1987 crash (Black Monday)",
        start="1987-10-13",
        end="1987-10-30",
        historical=True,
        summary=(
            "One-day Dow Jones Industrial Average decline on 19 October 1987, plus the "
            "four-session decline from 13 October. Equity path is the cited scalar, not a "
            "redistributed index tape. The H.15 10-year yield window is a public-domain path."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.226,
                "simple_return",
                "primary",
                "reported_one_decimal_percent",
                _BRADY
                + " Federal Reserve History states the same 22.6 percent one-day decline and the 508-point loss.",
                _FED_HISTORY_1987,
                _PD,
            ),
            _shock(
                "us_equity",
                -0.31,
                "simple_return",
                "context",
                "reported_integer_percent",
                _BRADY,
                _FED_HISTORY_1987,
                _PD,
            ),
            _shock(
                "us_equity",
                -0.046,
                "simple_return",
                "context",
                "reported_one_decimal_percent",
                "Federal Reserve History: by the end of 16 October 1987 the DJIA had lost 4.6 percent.",
                _FED_HISTORY_1987,
                _PD,
            ),
        ),
        window_id="crash_1987",
        limitations=(
            "The primary equity shock is the Dow Jones Industrial Average, not a total-return "
            "S&P 500 path. No intraday tape is bundled.",
            "A 10-year Treasury path is included from H.15; it is not an equity replay.",
        ),
    ),
    Crisis(
        crisis_id="dotcom_2000_02",
        name="2000–2002 dot-com unwind",
        start="2000-03-01",
        end="2002-10-31",
        historical=True,
        summary=(
            "Calendar window covering the unwind. No equity-index peak-to-trough scalar is "
            "stored: this build did not verify one against a primary document, and vendor "
            "index histories are not redistributed. The H.15 10-year yield path is included."
        ),
        shocks=(),
        window_id="dotcom_2000_02",
        limitations=(
            "Equity-index level path and a citable peak-to-trough percent are not in this "
            "catalog. A user-supplied return cache can be replayed; it is not generated here.",
            "The bundled series is the constant-maturity 10-year Treasury yield only.",
        ),
    ),
    Crisis(
        crisis_id="gfc_2008",
        name="2008 global financial crisis",
        start="2007-10-01",
        end="2009-03-31",
        historical=True,
        summary=(
            "Peak-to-trough equity decline stated by Federal Reserve History, with H.15 "
            "yield and effective federal funds paths. Not the DFAST hypothetical scenario."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.57,
                "simple_return",
                "primary",
                "reported_rounded_percent",
                "Federal Reserve History, The Great Recession: the S&P 500 fell 57 percent from its October 2007 peak to its March 2009 trough.",
                _GREAT_RECESSION,
                _PD,
            ),
            _shock(
                "house_price",
                -0.30,
                "simple_return",
                "context",
                "approximate",
                "Federal Reserve History, The Great Recession: home prices fell approximately 30 percent from their mid-2006 peak to mid-2009.",
                _GREAT_RECESSION,
                _PD,
            ),
        ),
        window_id="gfc_2008",
        limitations=(
            "The equity figure is a rounded peak-to-trough percent from a Fed essay, not a daily path.",
            "The essay also describes the funds-rate target moving from 5.25 percent in September 2007 to a 0–0.25 percent range in December 2008. The bundled path is the effective federal funds rate (DFF), which is a different series from the target range.",
        ),
    ),
    Crisis(
        crisis_id="flash_2010",
        name="2010 Flash Crash",
        start="2010-05-06",
        end="2010-05-06",
        historical=True,
        summary=(
            "Intraday equity-market stress on 6 May 2010 from the SEC/CFTC staff report. "
            "A daily close does not represent this episode. The H.15 10-year yield that day is bundled."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.10,
                "simple_return",
                "primary",
                "reported_range_severe_end",
                "SEC/CFTC staff report: between 14:41 and 14:45 the broad markets reached intra-day lows of 9–10 percent. The engine applies the severe end of that stated range (−10 percent).",
                _FLASH,
                _PD,
            ),
            _shock(
                "spy",
                -0.06,
                "simple_return",
                "context",
                "at_least",
                "SEC/CFTC staff report: prices of SPY suffered a decline of over 6 percent in the same phase. Stored as a lower bound on the magnitude, not an exact print.",
                _FLASH,
                _PD,
            ),
            _shock(
                "vix",
                0.225,
                "simple_return",
                "context",
                "reported_one_decimal_percent",
                "SEC/CFTC staff report: by 14:30 the VIX was up 22.5 percent from the opening level. This is a change in the volatility index, not an equity return.",
                _FLASH,
                _PD,
            ),
            _shock(
                "emini_spx",
                1056.0,
                "index_level",
                "context",
                "exact_level",
                "SEC/CFTC staff report: the E-mini reached an intraday low of 1056. A level, not a return.",
                _FLASH,
                _PD,
            ),
        ),
        window_id="flash_2010",
        limitations=(
            "Daily H.15 yields cannot replay the intraday trough. The equity shock is the report's 9–10 percent intraday range, severe endpoint.",
            "The often-quoted Dow point decline is not used: the text extract verified for this build states the 9–10 percent broad-market intraday low, not a separate point figure.",
        ),
    ),
    Crisis(
        crisis_id="chf_2015",
        name="2015 CHF de-peg",
        start="2015-01-14",
        end="2015-01-16",
        historical=True,
        summary=(
            "Swiss National Bank discontinued the CHF 1.20 per euro minimum rate on 15 January 2015. "
            "The intraday day-low print is a published scalar. The H.10 noon buying rates are a separate public-domain path and are smaller than that intraday low."
        ),
        shocks=(
            _shock(
                "eurchf",
                -0.30,
                "simple_return",
                "primary",
                "published_day_low_vs_prior_day_low",
                "Bonadio, Fischer, Sauré (Swiss Journal of Economics and Statistics, 2025): on 15 January 2015 the franc's day low was 0.84 CHF per EUR, against a previous-day low of 1.20. Simple change 0.84/1.20 − 1 = −0.30. Not a close.",
                _SJES_URL,
                _SJES,
            ),
            _shock(
                "snb_sight_deposit",
                -0.50,
                "rate_change_pp",
                "context",
                "exact",
                "SNB press release, 15 January 2015: the minimum exchange rate of CHF 1.20 per euro was discontinued, and the sight-deposit rate was lowered by 0.5 percentage points to −0.75 percent.",
                _SNB,
                "Swiss National Bank press release.",
            ),
        ),
        window_id="chf_2015",
        limitations=(
            "Do not add the −30 percent day-low shock to the H.10 noon path. They are two measurements of the same event.",
            "H.10 rates are noon buying rates in New York, not the intraday low.",
        ),
    ),
    Crisis(
        crisis_id="etf_2015",
        name="August 2015 ETF dislocation",
        start="2015-08-24",
        end="2015-08-24",
        historical=True,
        summary=(
            "SEC research note on 24 August 2015. The primary equity scalar is SPY's decline from the prior close to the day's low. A separate context figure records how many ETPs fell 20 percent or more."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.078,
                "simple_return",
                "primary",
                "reported_one_decimal_percent",
                "SEC research note: SPY's daily low on 24 August 2015 was 7.8 percent below the previous close, by 09:35. The note's instrument is SPY, used here as the equity-market stress scalar for that open.",
                _AUG24,
                _PD,
            ),
            _shock(
                "us_equity",
                -0.052,
                "simple_return",
                "context",
                "reported_one_decimal_percent",
                "SEC research note: SPY opened 5.2 percent below the previous close.",
                _AUG24,
                _PD,
            ),
            _shock(
                "us_equity",
                -0.042,
                "simple_return",
                "context",
                "reported_one_decimal_percent",
                "SEC research note: SPY closed down 4.2 percent.",
                _AUG24,
                _PD,
            ),
            _shock(
                "emini_spx",
                -0.05,
                "simple_return",
                "context",
                "exact",
                "SEC research note: the E-mini reached its limit-down price of 5 percent below the previous close and was paused from 09:25 to 09:30.",
                _AUG24,
                _PD,
            ),
            _shock(
                "etp_share_down_at_least_20pct",
                0.192,
                "fraction",
                "context",
                "exact",
                "SEC research note: 19.2 percent of ETPs declined by 20 percent or more, compared with 4.7 percent of corporate stocks. A cross-sectional share, not a portfolio return.",
                _AUG24,
                _PD,
            ),
        ),
        window_id="etf_2015",
        limitations=(
            "The 19.2 percent ETP share is not applied as a return. A book that held the dislocated tail is not identified by this scalar.",
            "The primary −7.8 percent figure is SPY's low, not the official cash-index close.",
        ),
    ),
    Crisis(
        crisis_id="volmageddon_2018",
        name="2018 Volmageddon",
        start="2018-02-05",
        end="2018-02-05",
        historical=True,
        summary=(
            "5 February 2018 equity decline and the inverse-volatility ETP loss, from the BIS Quarterly Review, March 2018. The VIX point jump is context. H.15 10-year yields for that week are bundled."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.042,
                "simple_return",
                "primary",
                "reported_one_decimal_percent",
                "BIS Quarterly Review, March 2018, box 'The equity market turbulence of 5 February': for the day as a whole the S&P 500 fell 4.2 percent. The same box also says the index fell 4 percent while the VIX jumped 20 points.",
                _BIS_URL,
                _BIS,
            ),
            _shock(
                "inverse_vol_etp",
                -0.84,
                "simple_return",
                "primary",
                "exact",
                "BIS Quarterly Review, March 2018: the value of one inverse-volatility ETP, XIV, fell 84 percent and the product was subsequently terminated.",
                _BIS_URL,
                _BIS,
            ),
            _shock(
                "vix",
                20.0,
                "level_change",
                "context",
                "exact",
                "BIS Quarterly Review, March 2018: on 5 February the VIX jumped 20 points, the largest daily increase since 1987. A point change, not a return.",
                _BIS_URL,
                _BIS,
            ),
        ),
        window_id="volmageddon_2018",
        limitations=(
            "FOMC minutes of 20–21 March 2018 describe the 5 February VIX rise as the highest level since 2015 and attribute part of it to unwinding short-volatility strategies. They do not add a second numeric shock.",
            "XIV's −84 percent is not an equity-index return. It is applied only to an inverse_vol_etp weight.",
        ),
    ),
    Crisis(
        crisis_id="covid_2020",
        name="2020 COVID crash",
        start="2020-02-14",
        end="2020-03-31",
        historical=True,
        summary=(
            "Policy-rate response and the H.15 Treasury and effective-funds paths through March 2020. "
            "An equity-index peak-to-trough percent was not verified from a primary document in this build and is not invented."
        ),
        shocks=(
            _shock(
                "fed_funds_target",
                -1.50,
                "rate_change_pp",
                "context",
                "exact",
                "Clarida, Duygan-Bump, Scotti, FEDS 2021-035: on 3 March and 15 March 2020 the FOMC cut the funds-rate target by 1.5 percentage points in total, to a 0–0.25 percent range.",
                _FEDS_COVID,
                _PD,
            ),
        ),
        window_id="covid_2020",
        limitations=(
            "No S&P or VIX path is bundled. Vendor index histories are not redistributed, and this build does not substitute an unverified peak-to-trough percent.",
            "The bundled DFF path is the effective rate. The −1.50 percentage-point figure is the target-range change and is context, not added to DFF.",
        ),
    ),
    Crisis(
        crisis_id="rates_2022",
        name="2022 rate shock",
        start="2022-01-03",
        end="2022-12-30",
        historical=True,
        summary=(
            "The 2022 tightening, measured by the bundled H.15 2-year, 10-year, and effective federal funds paths. "
            "The December 2022 target range is a cited level. An equity drawdown percent is not bundled."
        ),
        shocks=(
            _shock(
                "fed_funds_target",
                4.50,
                "index_level",
                "context",
                "target_range_upper",
                "FOMC statement, 14 December 2022: the Committee raised the target range for the federal funds rate to 4-1/4 to 4-1/2 percent. Stored as the top of that range, in percent, not as a return.",
                _FOMC_2022,
                _PD,
            ),
        ),
        window_id="rates_2022",
        limitations=(
            "The stress on a long bond book comes from the H.15 yield path (the largest yield increase inside the window), not from the target-range level.",
            "No equity-index drawdown is included. One was not verified here, and the index is not a public-domain series.",
        ),
    ),
    Crisis(
        crisis_id="dfast_2025_severely_adverse",
        name="DFAST 2025 severely adverse (hypothetical)",
        start="2024-10-01",
        end="2026-12-31",
        historical=False,
        summary=(
            "Board of Governors supervisory scenario published February 2025. Hypothetical. "
            "Not a forecast, not a historical replay, and not a substitute for the 2008 path."
        ),
        shocks=(
            _shock(
                "us_equity",
                -0.50,
                "simple_return",
                "primary",
                "exact",
                "Federal Reserve 2025 stress-test scenarios: equity prices fall 50 percent from 2024Q4 through 2025Q4.",
                _DFAST,
                _PD,
            ),
            _shock(
                "house_price",
                -0.33,
                "simple_return",
                "primary",
                "approximate",
                "Federal Reserve 2025 stress-test scenarios: house prices reach a trough about 33 percent below their 2024Q4 level.",
                _DFAST,
                _PD,
            ),
            _shock(
                "cre_price",
                -0.30,
                "simple_return",
                "primary",
                "exact",
                "Federal Reserve 2025 stress-test scenarios: commercial real estate prices trough 30 percent below their end-2024 level.",
                _DFAST,
                _PD,
            ),
            _shock(
                "vix",
                65.0,
                "index_level",
                "context",
                "exact_level",
                "Federal Reserve 2025 stress-test scenarios: the VIX, as the highest daily close in the quarter, peaks at 65 in 2025Q2. A level, not a return.",
                _DFAST,
                _PD,
            ),
            _shock(
                "unemployment_rate",
                5.9,
                "rate_change_pp",
                "context",
                "exact",
                "Federal Reserve 2025 stress-test scenarios: the unemployment rate rises 5.9 percentage points, from 4.1 percent in 2024Q4 to 10 percent.",
                _DFAST,
                _PD,
            ),
        ),
        window_id=None,
        limitations=(
            "Regulatory hypothetical. Excluded from the historical-crisis Mahalanobis cloud.",
            "The 2024 severely adverse scenario's 55 percent equity decline is a different year's scenario and is not applied here.",
        ),
    ),
)


_REQUIRED_IDS = (
    "crash_1987",
    "dotcom_2000_02",
    "gfc_2008",
    "flash_2010",
    "chf_2015",
    "etf_2015",
    "volmageddon_2018",
    "covid_2020",
    "rates_2022",
)


def crisis_by_id(crisis_id: str) -> Crisis:
    """Return one catalog entry. Unknown ids fail closed."""
    for crisis in CRISIS_CATALOG:
        if crisis.crisis_id == crisis_id:
            return crisis
    known = ", ".join(crisis.crisis_id for crisis in CRISIS_CATALOG)
    raise KeyError(f"unknown crisis_id {crisis_id!r}; known: {known}")


def require_historical_episodes() -> None:
    """Raise if a required historical episode was dropped from the catalog."""
    ids = {crisis.crisis_id for crisis in CRISIS_CATALOG}
    missing = [name for name in _REQUIRED_IDS if name not in ids]
    if missing:
        raise RuntimeError(f"crisis catalog is missing required episodes: {missing}")
