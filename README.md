# SIG Predictions Cup 2026: Automated trading bot

The SIG Predictions Cup was a trading competition organised by Susquehanna between October 1 and November 4, 2026. Each participant received 100,000 SUSQ of play money to trade 237 contracts on the US midterm elections, such as "Will the Democratic Party win the Arizona governorship?". Whoever had the most money after all races were decided won.

I'm a second-year economics student and this was my first trading competition. I built a bot that compared every contract with the prices on Polymarket and Kalshi, bought what looked too cheap and held it until the race was called. It ran on its own for five weeks without a single crash and ended up **3.96%** above where it started.

That's a modest result, and this write-up doesn't dress it up. The bot did almost exactly what my research said it would. The more useful part is what the research didn't see, which is the last section.

| | |
|---|---|
| Period | 1 October – 4 November 2026 |
| Contracts | 237 binary contracts across 117 races |
| Starting capital | 100,000 SUSQ |
| Account value, 8 November | **103,960 SUSQ (+3.96%)** |
| Cash / open positions | 102,091 / 1,869 SUSQ (12 positions still settling) |
| Crash restarts | 0 |

## The competition

Each contract paid 1 if the event happened and 0 if it didn't. Prices moved in half-cent steps between 0.5c and 99.5c, and the final ranking was the settled balance. Only the top three won prizes.

The exchange's per-account API limits shaped the whole design. A full refresh of every order book would have taken almost three minutes of reads, so the bot had to be selective about what it looked at and when.

## The strategy

Polymarket and Kalshi list the same races, with real money and far more traders. My starting assumption was that their prices are a decent estimate of the true probability, and that a new competition exchange would drift away from them. So the bot looked for SIG contracts trading below the external price and bought them:

```
edge   = fair − margin − ask
margin = max(0.01, external_spread / 2)
```

The margin made it conservative: the wider the external market, the less I trusted its price.

| Price | Minimum edge to buy |
|---|---|
| up to 93c | 2c |
| above 93c | 4c |

The higher bar above 93c is there because being slightly wrong about the fair value costs most of what such a contract can still pay. Positions were held to settlement. The bot sold early only if SIG's bid climbed back up to the fair value, because at that point there was nothing left to earn by holding.

## Running it safely

I decided early on that the bot would run from a server in the US. I'd never deployed anything to the cloud before, so the week before the open went mostly into learning SSH and systemd, including once locking myself out of my own machine.

What I cared about most was that the bot couldn't hurt me while I was asleep or at lectures. Every order path had a `DRY_RUN` flag that logged what it would have done instead of doing it, and every new feature ran that way first. Stopping the process, by hand or by the system, cancelled every open order before it exited. Rate limits had their own counters set below the exchange's, rejected requests were retried with growing pauses, and timeouts or server errors were retried instead of crashing the loop. By the end there were 657 tests.

The bot ran from the open on October 1 until a timer stopped it on November 3 and never restarted because of a crash.

## Sizing and limits

Position size started at a quarter of the Kelly fraction and then passed through a set of limits. The tightest one decided.

| Limit | Value |
|---|---|
| Kelly fraction | 0.25 |
| Per contract | 2% of capital (5% on the first day) |
| Per race, worst case | 2% |
| Per group of related races | 3% |
| Total exposure | 30%, raised to 40% and then 45% |

The per-race and group limits went live on the evening of October 1, a few hours too late. In the first hours the bot had already put large positions into Rhode Island Senate, New Hampshire's 1st district and Florida, with only the 5% per-contract limit in place. Those three races held about a third of everything I'd bought for the rest of the competition, with up to 9.9k at risk in Rhode Island alone. The new limits stopped the bot from adding to them, but nothing made it sell. I spent a day convinced the orders had somehow slipped past the limits before realising the limits simply didn't exist yet when they were placed.

## Three quiet weeks

After the first few days the bot mostly sat still. A typical log line was `targets=64 planned=0`: it already held what it wanted and couldn't find anything new worth buying.

I tried to understand why, and every test gave the same answer. I wasn't short of capital, I was short of mispriced contracts. A simulation without any exposure limit topped out at about 3,600 SUSQ of expected profit for the whole competition. Raising the limit from 45% to 60% would have added about 1,000. Trades with only 2–4c of edge were worth roughly nothing. Meanwhile, about half the account sat in cash.

## Research

Most of the code I wrote after the first week was research. Every script was read-only, with a guard that threw an error on any attempt to trade, stayed under 10 reads a minute so it couldn't slow down the live bot, and saved everything it fetched. Before looking at the data I wrote down what would count as a pass, which kept me honest more than once.

| Question | Answer |
|---|---|
| Were opening prices badly set? | No, they were close to the external markets right from the start. |
| Is there a pattern in who trades and when? | 388 hypotheses tested, nothing that moved the price. |
| Do prices within one race add up? | Often not, but the total opportunity was only 100–250 SUSQ. |
| Is SIG biased compared to the external markets? | Yes: favourites too cheap, longshots too expensive. |
| Would a stricter 4c filter on every buy help? | No, it cost 47 SUSQ in a replay, so I dropped it. |

The bias was good news, because it's exactly what the bot was already buying. The capacity result was the bad news: however much money I gave this strategy, it was never going to make much more.

## Market making

Carry only pays at the end, so in the second half of October I looked at the one strategy that could earn every day. The trade history suggested a market maker would have made 0.3–0.5 ticks per trade, even after accounting for being traded against by better-informed participants. Two days of recorded quotes agreed. Then I checked how much of that my quoter could actually reach.

| Step | Result |
|---|---|
| Edge per trade in the trade history | 0.3–0.5 ticks |
| Markets with a one-tick spread | 172 of 237 |
| Joining the best price | about +41 SUSQ a day at best |
| Improving the best price by one tick | about −23 SUSQ a day |
| Live trial on a wider set of markets | 0 quotes placed |

The edge was real, but it sat at the best price, where a large sponsor order was already waiting. I turned market making off on October 28.

## The end

On October 19 I set a timer to stop the bot at noon UTC on November 3, before the polls closed. It had no logic for results night, its prices would lag the real calls, and a slow bot in a fast market sounded like a good way to lose money.

On November 2 the account dropped by almost 3,000 SUSQ in a day without a single trade, because SIG prices on my biggest positions slipped below the external ones. I priced selling half of Rhode Island: it would have cost about 0.8k of expected value to cut the worst case from −9.9k to −5.6k. I kept everything, and none of the large losses happened.

On October 19 the positions I held implied a final account value of roughly 102,900 SUSQ, using the external prices as the true probabilities. The account stands at 103,960. The difference of about 1,060 comes from trades after that date, a few races settling slightly better than priced, and the 12 positions still waiting to settle. Within that margin, the strategy delivered what the research said it would.

## What I missed

Looking back, almost everything I built asked one question: how far is SIG's price from fair value? The carry engine bought that gap, the race check looked for it inside a race, the quoter tried to earn it at the best price and the edge filter tried to sharpen it. The measurements were right. The problem was what I chose to measure.

Some of the clearest leads were in my own notes from the first week. One was an exchange rule: an order waiting in the book sets no cash aside. I filed it under risks to watch, not as something to use. I also had a ladder of deep bids in the code, 7 to 50 cents below the market. It stayed switched off for the whole competition, and I never measured what it would have caught, because my research only used trades that had actually happened.

Capital was the second gap. For most of October about half the account sat in cash, because the carry engine could not find enough mispriced contracts to use it. I treated that as a limit of the strategy instead of a question to work on.

I also played it too safe for the game I was in. The money wasn't real and only the top three won anything, yet I built my limits as if it were my own savings, and I stopped the bot before results night, the one night when SIG's prices were most likely to lag. I had sketched a test of how quickly SIG followed the external markets, but I never built it: I hadn't been saving the external prices over time, so there was nothing to test it on.

| I'd keep | I'd change |
|---|---|
| Dry run first, a working kill switch, tests | Record every external price from day one, so later questions can be tested |
| Writing down the pass rule before the data | Run cheap experiments live at small size instead of proving them offline |
| A bot that runs for weeks untouched | Treat idle cash as a problem to solve, not a safety margin |
| Conservative fair values | Size risk for the prize structure, not for real savings |

## About this repository

This repository holds the bot that ran during the competition and this write-up. The code was written for one exchange and one five-week event, and that API no longer exists, so it can't be run as it is. Credentials, raw exchange data and server configuration are not included.

| File | Role |
|---|---|
| `main.py` | trading cycle, scheduling and graceful shutdown |
| `config.py` | validated settings with safe defaults |
| `logging_config.py` | structured logging to console and a rotating file |
| `oracle_api.py` | Polymarket and Kalshi order-book quotes |
| `market_mapper.py` | mapping between SIG contracts and external markets |
| `exchange_api.py` | exchange client with rate limits and retries |
| `strategy.py` | fair value from the external quotes, edges, targets and the buy/sell list |
| `risk.py` | fractional Kelly sizing and limits |
| `fills.py` | fill prices per data source (a NO fill is quoted in NO terms) |
| `markout.py` | post-trade markout |
| `quoter.py` | two-sided quoter, kept off in production |
| `ladder.py` | deep-bid ladder, kept off in production |
| `edge_scan.py` | read-only scan of past liquidity holes and D-vs-R price gaps |
| `research/` | shared helpers for the research scripts: read-only API access, statistics, memory guard |
| `scripts/` | read-only analysis and reporting tools, plus one guarded live test (`reserve_test.py`) |
| `tests/` | unit and regression tests |
| `requirements.txt` | Python dependencies |

The code is released under the [MIT License](LICENSE). Built for a competition with play money; nothing here is investment advice.
