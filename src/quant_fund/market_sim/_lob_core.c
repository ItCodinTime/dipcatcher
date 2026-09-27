/*
 * Single-instrument price-time priority matching core.
 *
 * Integer ticks and share quantities. One Book has no process-global
 * matching state. Not thread-safe on a single Book. Simulation only:
 * nothing here routes orders to a broker.
 *
 * Limit prices are integers in [1, LOB_PRICE_MAX). Price LOB_PRICE_MAX is
 * the market-buy sentinel used only while halted (market-on-open). Price 0
 * is the market-sell sentinel. Continuous trading never rests at either.
 *
 * Time priority is insertion order at a price. Callers attach nanosecond
 * timestamps; the engine stores them and does not read a wall clock except
 * inside lob_bench.
 */

#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define LOB_PRICE_MAX 100000
#define LOB_MAX_ORDERS (1 << 20)
#define LOB_NIL (-1)

enum {
    EV_LIMIT = 1,
    EV_MARKET = 2,
    EV_CANCEL = 3,
    EV_IOC = 4,
    EV_FOK = 5
};

enum {
    ST_REJECTED = 1,
    ST_FILLED = 2,
    ST_PARTIAL_REST = 3,
    ST_RESTING = 4,
    ST_CANCELLED = 5,
    ST_IOC_DONE = 6,
    ST_FOK_UNFILLED = 7
};

enum {
    REJ_NONE = 0,
    REJ_QTY = 1,
    REJ_PRICE = 2,
    REJ_FULL = 3,
    REJ_UNKNOWN = 4,
    REJ_HALT_IMMEDIATE = 5,
    REJ_NOT_HALTED = 6,
    REJ_BAD_TYPE = 7
};

enum { TIF_GTC = 0, TIF_IOC = 1, TIF_MOO = 2 };

typedef struct {
    int64_t ts;
    int32_t price;
    int32_t qty;
    int32_t aggressor_side;
    int32_t taker_agent;
    int32_t maker_agent;
    int32_t _pad;
    int64_t taker_order_id;
    int64_t maker_order_id;
} Trade;

typedef struct {
    int64_t ts;
    int32_t type;
    int32_t side;
    int32_t price;
    int32_t qty;
    int32_t agent;
    int32_t _pad;
    int64_t order_id;
} Event;

typedef struct {
    int32_t status;
    int32_t reason;
    int32_t filled_qty;
    int32_t resting_qty;
    int32_t n_trades;
    int32_t truncated;
    int64_t order_id;
    int64_t filled_notional;
} EventResult;

typedef struct {
    int32_t price;
    int32_t n_trades;
    int32_t n_moo_cancelled;
    int32_t truncated;
    int32_t status;
    int32_t _pad;
} UncrossResult;

typedef struct {
    int64_t bid_qty;
    int64_t ask_qty;
    int64_t trade_qty;
    int64_t trade_notional;
    int64_t checksum;
    int32_t n_bid_orders;
    int32_t n_ask_orders;
    int32_t n_live;
    int32_t best_bid;
    int32_t best_ask;
    int32_t crossed;
    int32_t halted;
    int32_t n_trades;
    int32_t list_ok;
    int32_t _pad;
} Audit;

typedef struct {
    int64_t n_events;
    int64_t n_trades;
    int64_t checksum;
    int64_t elapsed_match_ns;
    int64_t elapsed_generate_ns;
    int32_t list_ok;
    int32_t n_rejects;
    int32_t _pad;
} BenchResult;

typedef struct {
    int64_t ts;
    int32_t price;
    int32_t qty;
    int32_t agent;
    int32_t next;
    int32_t prev;
    int32_t gen;
    int32_t side;
    int32_t alive;
    int32_t tif;
} Order;

typedef struct Book {
    uint32_t magic;
    int32_t halted;
    int32_t record;
    int32_t truncated;
    int32_t best_bid;
    int32_t best_ask;
    int32_t n_high;
    int32_t n_live;
    int32_t free_head;
    int32_t log_n;
    int32_t log_cap;
    int32_t last_trade_px;
    int32_t n_trades;
    int64_t trade_qty;
    int64_t trade_notional;
    uint64_t checksum;
    Order *orders;
    int32_t *bid_head;
    int32_t *bid_tail;
    int32_t *ask_head;
    int32_t *ask_tail;
    int64_t *bid_qty;
    int64_t *ask_qty;
    /* Occupied-price chains so the touch moves in O(levels), not O(empty ticks). */
    int32_t *bid_up;
    int32_t *bid_dn;
    int32_t *ask_up;
    int32_t *ask_dn;
    Trade *log;
} Book;

#define LOB_MAGIC 0x4C4F4231u

_Static_assert(sizeof(Trade) == 48, "Trade layout drifted");
_Static_assert(sizeof(Event) == 40, "Event layout drifted");
_Static_assert(sizeof(EventResult) == 40, "EventResult layout drifted");
_Static_assert(sizeof(Order) == 48, "Order layout drifted");

const char *lob_version(void) { return "lob-core-1"; }

int32_t lob_price_max(void) { return LOB_PRICE_MAX; }

int32_t lob_max_orders(void) { return LOB_MAX_ORDERS; }

int32_t lob_sizeof_trade(void) { return (int32_t)sizeof(Trade); }

int32_t lob_sizeof_event(void) { return (int32_t)sizeof(Event); }

int32_t lob_sizeof_result(void) { return (int32_t)sizeof(EventResult); }

int32_t lob_sizeof_uncross(void) { return (int32_t)sizeof(UncrossResult); }

int32_t lob_sizeof_audit(void) { return (int32_t)sizeof(Audit); }

int32_t lob_sizeof_bench(void) { return (int32_t)sizeof(BenchResult); }

static int64_t make_id(int32_t slot, int32_t gen) {
    return ((int64_t)gen << 20) | (uint32_t)slot;
}

static int resolve_id(const Book *b, int64_t id, int32_t *slot_out) {
    int32_t slot = (int32_t)(id & ((1 << 20) - 1));
    int32_t gen = (int32_t)(id >> 20);
    if (slot < 0 || slot >= b->n_high) {
        return 0;
    }
    const Order *o = &b->orders[slot];
    if (!o->alive || o->gen != gen) {
        return 0;
    }
    *slot_out = slot;
    return 1;
}

static int real_price(int32_t price) { return price >= 1 && price <= LOB_PRICE_MAX - 1; }

static void insert_bid_level(Book *b, int32_t price) {
    b->bid_up[price] = LOB_NIL;
    b->bid_dn[price] = LOB_NIL;
    if (b->best_bid < 0) {
        b->best_bid = price;
        return;
    }
    if (price > b->best_bid) {
        b->bid_dn[price] = b->best_bid;
        b->bid_up[b->best_bid] = price;
        b->best_bid = price;
        return;
    }
    int32_t above = b->best_bid;
    int guard = 0;
    while (b->bid_dn[above] >= 0 && b->bid_dn[above] > price && guard++ < LOB_PRICE_MAX) {
        above = b->bid_dn[above];
    }
    int32_t below = b->bid_dn[above];
    b->bid_up[price] = above;
    b->bid_dn[price] = below;
    b->bid_dn[above] = price;
    if (below >= 0) {
        b->bid_up[below] = price;
    }
}

static void insert_ask_level(Book *b, int32_t price) {
    b->ask_up[price] = LOB_NIL;
    b->ask_dn[price] = LOB_NIL;
    if (b->best_ask < 0) {
        b->best_ask = price;
        return;
    }
    if (price < b->best_ask) {
        b->ask_up[price] = b->best_ask;
        b->ask_dn[b->best_ask] = price;
        b->best_ask = price;
        return;
    }
    int32_t below = b->best_ask;
    int guard = 0;
    while (b->ask_up[below] >= 0 && b->ask_up[below] < price && guard++ < LOB_PRICE_MAX) {
        below = b->ask_up[below];
    }
    int32_t above = b->ask_up[below];
    b->ask_dn[price] = below;
    b->ask_up[price] = above;
    b->ask_up[below] = price;
    if (above >= 0) {
        b->ask_dn[above] = price;
    }
}

static void unlink_bid_level(Book *b, int32_t price) {
    int32_t up = b->bid_up[price];
    int32_t dn = b->bid_dn[price];
    if (up >= 0) {
        b->bid_dn[up] = dn;
    }
    if (dn >= 0) {
        b->bid_up[dn] = up;
    }
    if (b->best_bid == price) {
        b->best_bid = dn;
    }
    b->bid_up[price] = LOB_NIL;
    b->bid_dn[price] = LOB_NIL;
}

static void unlink_ask_level(Book *b, int32_t price) {
    int32_t up = b->ask_up[price];
    int32_t dn = b->ask_dn[price];
    if (up >= 0) {
        b->ask_dn[up] = dn;
    }
    if (dn >= 0) {
        b->ask_up[dn] = up;
    }
    if (b->best_ask == price) {
        b->best_ask = up;
    }
    b->ask_up[price] = LOB_NIL;
    b->ask_dn[price] = LOB_NIL;
}

static void note_add_qty(Book *b, int32_t side, int32_t price, int32_t qty, int update_touch) {
    if (side > 0) {
        int64_t before = b->bid_qty[price];
        b->bid_qty[price] = before + qty;
        if (before == 0 && update_touch && real_price(price)) {
            insert_bid_level(b, price);
        }
    } else {
        int64_t before = b->ask_qty[price];
        b->ask_qty[price] = before + qty;
        if (before == 0 && update_touch && real_price(price)) {
            insert_ask_level(b, price);
        }
    }
}

static void note_sub_qty(Book *b, int32_t side, int32_t price, int32_t qty) {
    if (side > 0) {
        b->bid_qty[price] -= qty;
        if (b->bid_qty[price] == 0 && real_price(price)) {
            unlink_bid_level(b, price);
        }
    } else {
        b->ask_qty[price] -= qty;
        if (b->ask_qty[price] == 0 && real_price(price)) {
            unlink_ask_level(b, price);
        }
    }
}

static void recompute_touch(Book *b) {
    int n = LOB_PRICE_MAX + 1;
    memset(b->bid_up, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->bid_dn, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->ask_up, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->ask_dn, 0xFF, (size_t)n * sizeof(int32_t));
    b->best_bid = LOB_NIL;
    b->best_ask = LOB_NIL;
    int32_t last_bid = LOB_NIL;
    for (int p = 1; p <= LOB_PRICE_MAX - 1; p++) {
        if (b->bid_qty[p] > 0 && b->bid_head[p] >= 0) {
            b->bid_dn[p] = last_bid;
            b->bid_up[p] = LOB_NIL;
            if (last_bid >= 0) {
                b->bid_up[last_bid] = p;
            }
            last_bid = p;
        }
    }
    b->best_bid = last_bid;
    int32_t last_ask = LOB_NIL;
    for (int p = 1; p <= LOB_PRICE_MAX - 1; p++) {
        if (b->ask_qty[p] > 0 && b->ask_head[p] >= 0) {
            if (b->best_ask < 0) {
                b->best_ask = p;
            }
            b->ask_dn[p] = last_ask;
            b->ask_up[p] = LOB_NIL;
            if (last_ask >= 0) {
                b->ask_up[last_ask] = p;
            }
            last_ask = p;
        }
    }
}

static void free_slot(Book *b, int32_t slot) {
    Order *o = &b->orders[slot];
    o->alive = 0;
    o->qty = 0;
    o->gen += 1;
    if (o->gen <= 0) {
        o->gen = 1;
    }
    o->next = b->free_head;
    o->prev = LOB_NIL;
    b->free_head = slot;
    b->n_live--;
}

static void unlink_empty(Book *b, int32_t slot) {
    Order *o = &b->orders[slot];
    int32_t price = o->price;
    int32_t side = o->side;
    int32_t *head = (side > 0) ? b->bid_head : b->ask_head;
    int32_t *tail = (side > 0) ? b->bid_tail : b->ask_tail;
    int64_t level = (side > 0) ? b->bid_qty[price] : b->ask_qty[price];
    if (o->prev >= 0) {
        b->orders[o->prev].next = o->next;
    } else {
        head[price] = o->next;
    }
    if (o->next >= 0) {
        b->orders[o->next].prev = o->prev;
    } else {
        tail[price] = o->prev;
    }
    if (level <= 0) {
        head[price] = LOB_NIL;
        tail[price] = LOB_NIL;
    }
}

static void remove_live(Book *b, int32_t slot) {
    Order *o = &b->orders[slot];
    int32_t qty = o->qty;
    int32_t price = o->price;
    int32_t side = o->side;
    note_sub_qty(b, side, price, qty);
    o->qty = 0;
    unlink_empty(b, slot);
    free_slot(b, slot);
}

static int32_t alloc_slot(Book *b) {
    int32_t slot;
    if (b->free_head >= 0) {
        slot = b->free_head;
        b->free_head = b->orders[slot].next;
    } else {
        if (b->n_high >= LOB_MAX_ORDERS) {
            return LOB_NIL;
        }
        slot = b->n_high++;
        b->orders[slot].gen = 1;
    }
    b->n_live++;
    return slot;
}

static void record_trade(
    Book *b,
    int64_t ts,
    int32_t price,
    int32_t qty,
    int32_t side,
    int32_t taker,
    int32_t maker,
    int64_t taker_id,
    int64_t maker_id
) {
    b->n_trades++;
    b->trade_qty += qty;
    b->trade_notional += (int64_t)price * (int64_t)qty;
    b->last_trade_px = price;
    b->checksum = b->checksum * 1315423911ULL + (uint64_t)price * 17ULL + (uint64_t)qty +
                  (uint64_t)(side + 3) * 131ULL + (uint64_t)(taker + 17) +
                  ((uint64_t)maker << 8);
    if (!b->record) {
        return;
    }
    if (b->log_n >= b->log_cap) {
        int32_t cap = b->log_cap > 0 ? b->log_cap * 2 : 256;
        Trade *grown = (Trade *)realloc(b->log, (size_t)cap * sizeof(Trade));
        if (grown == NULL) {
            b->truncated = 1;
            return;
        }
        b->log = grown;
        b->log_cap = cap;
    }
    Trade *t = &b->log[b->log_n++];
    t->ts = ts;
    t->price = price;
    t->qty = qty;
    t->aggressor_side = side;
    t->taker_agent = taker;
    t->maker_agent = maker;
    t->_pad = 0;
    t->taker_order_id = taker_id;
    t->maker_order_id = maker_id;
}

static void reduce_order(Book *b, int32_t slot, int32_t fill) {
    Order *o = &b->orders[slot];
    int32_t price = o->price;
    int32_t side = o->side;
    o->qty -= fill;
    note_sub_qty(b, side, price, fill);
    if (o->qty == 0) {
        unlink_empty(b, slot);
        free_slot(b, slot);
    }
}

static int64_t link_new(
    Book *b,
    int32_t side,
    int32_t price,
    int32_t qty,
    int64_t ts,
    int32_t agent,
    int32_t tif,
    int update_best
) {
    int32_t slot = alloc_slot(b);
    if (slot < 0) {
        return -1;
    }
    Order *o = &b->orders[slot];
    o->ts = ts;
    o->price = price;
    o->qty = qty;
    o->agent = agent;
    o->side = side;
    o->alive = 1;
    o->tif = tif;
    o->next = LOB_NIL;
    o->prev = (side > 0) ? b->bid_tail[price] : b->ask_tail[price];
    if (o->prev >= 0) {
        b->orders[o->prev].next = slot;
    } else if (side > 0) {
        b->bid_head[price] = slot;
    } else {
        b->ask_head[price] = slot;
    }
    if (side > 0) {
        b->bid_tail[price] = slot;
    } else {
        b->ask_tail[price] = slot;
    }
    note_add_qty(b, side, price, qty, update_best);
    return make_id(slot, o->gen);
}

static int64_t available_qty(const Book *b, int32_t side, int32_t limit) {
    int64_t avail = 0;
    int guard = 0;
    if (side > 0) {
        if (b->best_ask < 0 || limit < b->best_ask) {
            return 0;
        }
        for (int p = b->best_ask; p >= 0 && p <= limit && guard++ < LOB_PRICE_MAX; p = b->ask_up[p]) {
            avail += b->ask_qty[p];
        }
    } else {
        if (b->best_bid < 0 || limit > b->best_bid) {
            return 0;
        }
        for (int p = b->best_bid; p >= limit && guard++ < LOB_PRICE_MAX; p = b->bid_dn[p]) {
            avail += b->bid_qty[p];
        }
    }
    return avail;
}

static int match_buy(
    Book *b, int32_t limit, int32_t qty, int64_t ts, int32_t agent, int64_t taker_id, int64_t *notional
) {
    int filled = 0;
    while (qty > 0 && b->best_ask >= 0 && b->best_ask <= limit) {
        int32_t slot = b->ask_head[b->best_ask];
        if (slot < 0) {
            break;
        }
        Order *m = &b->orders[slot];
        int32_t fill = m->qty < qty ? m->qty : qty;
        int32_t price = m->price;
        int32_t maker_agent = m->agent;
        int64_t maker_id = make_id(slot, m->gen);
        reduce_order(b, slot, fill);
        record_trade(b, ts, price, fill, 1, agent, maker_agent, taker_id, maker_id);
        *notional += (int64_t)price * (int64_t)fill;
        qty -= fill;
        filled += fill;
    }
    return filled;
}

static int match_sell(
    Book *b, int32_t limit, int32_t qty, int64_t ts, int32_t agent, int64_t taker_id, int64_t *notional
) {
    int filled = 0;
    while (qty > 0 && b->best_bid >= 0 && b->best_bid >= limit) {
        int32_t slot = b->bid_head[b->best_bid];
        if (slot < 0) {
            break;
        }
        Order *m = &b->orders[slot];
        int32_t fill = m->qty < qty ? m->qty : qty;
        int32_t price = m->price;
        int32_t maker_agent = m->agent;
        int64_t maker_id = make_id(slot, m->gen);
        reduce_order(b, slot, fill);
        record_trade(b, ts, price, fill, -1, agent, maker_agent, taker_id, maker_id);
        *notional += (int64_t)price * (int64_t)fill;
        qty -= fill;
        filled += fill;
    }
    return filled;
}

static void begin_mutation(Book *b) {
    b->log_n = 0;
    b->truncated = 0;
}

static void finish_result(
    EventResult *out,
    int32_t status,
    int32_t reason,
    int32_t filled,
    int32_t resting,
    int64_t order_id,
    int64_t notional,
    const Book *b
) {
    out->status = status;
    out->reason = reason;
    out->filled_qty = filled;
    out->resting_qty = resting;
    out->n_trades = b->log_n;
    out->truncated = b->truncated;
    out->order_id = order_id;
    out->filled_notional = notional;
}

static void reject(Book *b, EventResult *out, int32_t reason) {
    finish_result(out, ST_REJECTED, reason, 0, 0, 0, 0, b);
}

Book *lob_new(void) {
    Book *b = (Book *)calloc(1, sizeof(Book));
    if (b == NULL) {
        return NULL;
    }
    int n = LOB_PRICE_MAX + 1;
    b->orders = (Order *)calloc((size_t)LOB_MAX_ORDERS, sizeof(Order));
    b->bid_head = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->bid_tail = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->ask_head = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->ask_tail = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->bid_qty = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    b->ask_qty = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    b->bid_up = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->bid_dn = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->ask_up = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    b->ask_dn = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (b->orders == NULL || b->bid_head == NULL || b->bid_tail == NULL || b->ask_head == NULL ||
        b->ask_tail == NULL || b->bid_qty == NULL || b->ask_qty == NULL || b->bid_up == NULL ||
        b->bid_dn == NULL || b->ask_up == NULL || b->ask_dn == NULL) {
        free(b->orders);
        free(b->bid_head);
        free(b->bid_tail);
        free(b->ask_head);
        free(b->ask_tail);
        free(b->bid_qty);
        free(b->ask_qty);
        free(b->bid_up);
        free(b->bid_dn);
        free(b->ask_up);
        free(b->ask_dn);
        free(b);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        b->bid_head[i] = LOB_NIL;
        b->bid_tail[i] = LOB_NIL;
        b->ask_head[i] = LOB_NIL;
        b->ask_tail[i] = LOB_NIL;
    }
    memset(b->bid_up, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->bid_dn, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->ask_up, 0xFF, (size_t)n * sizeof(int32_t));
    memset(b->ask_dn, 0xFF, (size_t)n * sizeof(int32_t));
    b->magic = LOB_MAGIC;
    b->best_bid = LOB_NIL;
    b->best_ask = LOB_NIL;
    b->free_head = LOB_NIL;
    b->record = 1;
    b->last_trade_px = 0;
    return b;
}

void lob_free(Book *b) {
    if (b == NULL || b->magic != LOB_MAGIC) {
        return;
    }
    b->magic = 0;
    free(b->orders);
    free(b->bid_head);
    free(b->bid_tail);
    free(b->ask_head);
    free(b->ask_tail);
    free(b->bid_qty);
    free(b->ask_qty);
    free(b->bid_up);
    free(b->bid_dn);
    free(b->ask_up);
    free(b->ask_dn);
    free(b->log);
    free(b);
}

void lob_halt(Book *b, int32_t halted) {
    if (b == NULL || b->magic != LOB_MAGIC) {
        return;
    }
    b->halted = halted ? 1 : 0;
}

int32_t lob_order_qty(const Book *b, int64_t order_id) {
    int32_t slot;
    if (b == NULL || b->magic != LOB_MAGIC || !resolve_id(b, order_id, &slot)) {
        return -1;
    }
    return b->orders[slot].qty;
}

void lob_touch(const Book *b, int32_t *bid, int32_t *ask, int64_t *bid_qty, int64_t *ask_qty) {
    int32_t bb = (b != NULL) ? b->best_bid : LOB_NIL;
    int32_t ba = (b != NULL) ? b->best_ask : LOB_NIL;
    if (bid != NULL) {
        *bid = bb;
    }
    if (ask != NULL) {
        *ask = ba;
    }
    if (bid_qty != NULL) {
        *bid_qty = (bb >= 0) ? b->bid_qty[bb] : 0;
    }
    if (ask_qty != NULL) {
        *ask_qty = (ba >= 0) ? b->ask_qty[ba] : 0;
    }
}

int64_t lob_depth(const Book *b, int32_t side, int32_t n_levels) {
    if (b == NULL || b->magic != LOB_MAGIC || n_levels <= 0) {
        return 0;
    }
    int64_t sum = 0;
    int found = 0;
    if (side > 0) {
        for (int p = b->best_bid; p >= 1 && found < n_levels; p = b->bid_dn[p]) {
            sum += b->bid_qty[p];
            found++;
        }
    } else {
        for (int p = b->best_ask; p >= 1 && p < LOB_PRICE_MAX && found < n_levels; p = b->ask_up[p]) {
            sum += b->ask_qty[p];
            found++;
        }
    }
    return sum;
}

int64_t lob_level_qty(const Book *b, int32_t side, int32_t price) {
    if (b == NULL || price < 0 || price > LOB_PRICE_MAX) {
        return 0;
    }
    return (side > 0) ? b->bid_qty[price] : b->ask_qty[price];
}

const Trade *lob_trades(const Book *b, int32_t *n_out) {
    if (n_out != NULL) {
        *n_out = (b != NULL) ? b->log_n : 0;
    }
    if (b == NULL || b->log_n <= 0) {
        return NULL;
    }
    return b->log;
}

static int valid_limit_price(int32_t price) {
    return price >= 1 && price <= LOB_PRICE_MAX - 1;
}

static int valid_qty(int32_t qty) { return qty > 0 && qty <= 1000000000; }

void lob_submit(Book *b, const Event *ev, EventResult *out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (b == NULL || b->magic != LOB_MAGIC || ev == NULL) {
        out->status = ST_REJECTED;
        out->reason = REJ_BAD_TYPE;
        return;
    }
    begin_mutation(b);
    if (ev->type == EV_CANCEL) {
        int32_t slot;
        if (!resolve_id(b, ev->order_id, &slot)) {
            reject(b, out, REJ_UNKNOWN);
            return;
        }
        remove_live(b, slot);
        finish_result(out, ST_CANCELLED, REJ_NONE, 0, 0, ev->order_id, 0, b);
        return;
    }
    if (ev->type != EV_LIMIT && ev->type != EV_MARKET && ev->type != EV_IOC && ev->type != EV_FOK) {
        reject(b, out, REJ_BAD_TYPE);
        return;
    }
    if (ev->side != 1 && ev->side != -1) {
        reject(b, out, REJ_BAD_TYPE);
        return;
    }
    if (!valid_qty(ev->qty)) {
        reject(b, out, REJ_QTY);
        return;
    }
    int is_market = ev->type == EV_MARKET;
    int is_ioc = ev->type == EV_IOC || is_market;
    int is_fok = ev->type == EV_FOK;
    int32_t limit = ev->price;
    if (is_market) {
        limit = (ev->side > 0) ? (LOB_PRICE_MAX - 1) : 1;
    } else if (!valid_limit_price(limit)) {
        reject(b, out, REJ_PRICE);
        return;
    }
    if (b->halted) {
        /* Market orders rest as market-on-open. IOC and FOK do not. */
        if (!is_market && (is_ioc || is_fok)) {
            reject(b, out, REJ_HALT_IMMEDIATE);
            return;
        }
        int32_t price = limit;
        int32_t tif = TIF_GTC;
        int update_best = 1;
        if (is_market) {
            price = (ev->side > 0) ? LOB_PRICE_MAX : 0;
            tif = TIF_MOO;
            update_best = 0;
        }
        int64_t id = link_new(b, ev->side, price, ev->qty, ev->ts, ev->agent, tif, update_best);
        if (id < 0) {
            reject(b, out, REJ_FULL);
            return;
        }
        finish_result(out, ST_RESTING, REJ_NONE, 0, ev->qty, id, 0, b);
        return;
    }
    if (is_fok) {
        int64_t avail = available_qty(b, ev->side, limit);
        if (avail < (int64_t)ev->qty) {
            finish_result(out, ST_FOK_UNFILLED, REJ_NONE, 0, 0, 0, 0, b);
            return;
        }
    }
    int32_t slot = alloc_slot(b);
    if (slot < 0) {
        reject(b, out, REJ_FULL);
        return;
    }
    Order *taker = &b->orders[slot];
    taker->ts = ev->ts;
    taker->price = limit;
    taker->qty = ev->qty;
    taker->agent = ev->agent;
    taker->side = ev->side;
    taker->alive = 1;
    taker->tif = is_ioc ? TIF_IOC : TIF_GTC;
    taker->next = LOB_NIL;
    taker->prev = LOB_NIL;
    int64_t taker_id = make_id(slot, taker->gen);
    int64_t notional = 0;
    int filled;
    if (ev->side > 0) {
        filled = match_buy(b, limit, ev->qty, ev->ts, ev->agent, taker_id, &notional);
    } else {
        filled = match_sell(b, limit, ev->qty, ev->ts, ev->agent, taker_id, &notional);
    }
    int32_t left = ev->qty - filled;
    if (left > 0 && !is_ioc && !is_fok) {
        /* Slot is already allocated and still alive. Link the residual. */
        taker = &b->orders[slot];
        taker->qty = left;
        taker->price = limit;
        taker->prev = (ev->side > 0) ? b->bid_tail[limit] : b->ask_tail[limit];
        taker->next = LOB_NIL;
        if (taker->prev >= 0) {
            b->orders[taker->prev].next = slot;
        } else if (ev->side > 0) {
            b->bid_head[limit] = slot;
        } else {
            b->ask_head[limit] = slot;
        }
        if (ev->side > 0) {
            b->bid_tail[limit] = slot;
        } else {
            b->ask_tail[limit] = slot;
        }
        note_add_qty(b, ev->side, limit, left, 1);
        int32_t status = (filled > 0) ? ST_PARTIAL_REST : ST_RESTING;
        finish_result(out, status, REJ_NONE, filled, left, taker_id, notional, b);
        return;
    }
    /* Fully filled, or IOC/FOK residual is cancelled. The slot never rested. */
    free_slot(b, slot);
    if (is_fok && filled != ev->qty) {
        /* FOK was pre-checked; a shortfall here is an engine bug. Report it. */
        finish_result(out, ST_REJECTED, REJ_BAD_TYPE, filled, 0, 0, notional, b);
        return;
    }
    if (is_ioc || is_market) {
        finish_result(out, ST_IOC_DONE, REJ_NONE, filled, 0, taker_id, notional, b);
        return;
    }
    finish_result(out, ST_FILLED, REJ_NONE, filled, 0, taker_id, notional, b);
}

static int32_t seek_bid(const Book *b, int32_t from, int32_t pstar) {
    for (int p = from; p >= pstar; p--) {
        if (b->bid_head[p] >= 0) {
            return p;
        }
    }
    return LOB_NIL;
}

static int32_t seek_ask(const Book *b, int32_t from, int32_t pstar) {
    for (int p = from; p <= pstar; p++) {
        if (b->ask_head[p] >= 0) {
            return p;
        }
    }
    return LOB_NIL;
}

static void cancel_level(Book *b, int32_t side, int32_t price, int32_t *count) {
    int32_t *head = (side > 0) ? b->bid_head : b->ask_head;
    int guard = 0;
    while (head[price] >= 0 && guard++ < LOB_MAX_ORDERS) {
        remove_live(b, head[price]);
        *count += 1;
    }
}

void lob_uncross(Book *b, int64_t ts, int32_t ref_price, UncrossResult *out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (b == NULL || b->magic != LOB_MAGIC) {
        out->status = 2;
        return;
    }
    begin_mutation(b);
    if (!b->halted) {
        out->status = 1;
        out->truncated = 0;
        return;
    }
    int n = LOB_PRICE_MAX + 1;
    int64_t *buy_w = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *sell_w = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (buy_w == NULL || sell_w == NULL) {
        free(buy_w);
        free(sell_w);
        out->status = 2;
        return;
    }
    int64_t running = 0;
    for (int p = LOB_PRICE_MAX; p >= 1; p--) {
        running += b->bid_qty[p];
        buy_w[p] = running;
    }
    running = 0;
    for (int p = 0; p <= LOB_PRICE_MAX - 1; p++) {
        running += b->ask_qty[p];
        sell_w[p] = running;
    }
    int32_t ref = ref_price;
    if (ref <= 0) {
        ref = b->last_trade_px > 0 ? b->last_trade_px : 1;
    }
    int64_t best_vol = 0;
    int32_t best_p = LOB_NIL;
    int best_dist = 0;
    for (int p = 1; p <= LOB_PRICE_MAX - 1; p++) {
        int64_t vol = buy_w[p] < sell_w[p] ? buy_w[p] : sell_w[p];
        if (vol <= 0) {
            continue;
        }
        int dist = p > ref ? p - ref : ref - p;
        if (best_p < 0 || vol > best_vol ||
            (vol == best_vol && (dist < best_dist || (dist == best_dist && p < best_p)))) {
            best_vol = vol;
            best_p = p;
            best_dist = dist;
        }
    }
    free(buy_w);
    free(sell_w);
    if (best_p < 0 || best_vol <= 0) {
        cancel_level(b, 1, LOB_PRICE_MAX, &out->n_moo_cancelled);
        cancel_level(b, -1, 0, &out->n_moo_cancelled);
        b->halted = 0;
        recompute_touch(b);
        out->status = 0;
        out->price = 0;
        out->n_trades = 0;
        out->truncated = b->truncated;
        return;
    }
    int32_t b_price = seek_bid(b, LOB_PRICE_MAX, best_p);
    int32_t s_price = seek_ask(b, 0, best_p);
    int32_t bslot = (b_price >= 0) ? b->bid_head[b_price] : LOB_NIL;
    int32_t sslot = (s_price >= 0) ? b->ask_head[s_price] : LOB_NIL;
    int guard = 0;
    while (bslot >= 0 && sslot >= 0 && guard++ < LOB_MAX_ORDERS) {
        int32_t bnext = b->orders[bslot].next;
        int32_t snext = b->orders[sslot].next;
        int32_t fill = b->orders[bslot].qty < b->orders[sslot].qty ? b->orders[bslot].qty
                                                                    : b->orders[sslot].qty;
        int32_t buy_agent = b->orders[bslot].agent;
        int32_t sell_agent = b->orders[sslot].agent;
        int64_t buy_id = make_id(bslot, b->orders[bslot].gen);
        int64_t sell_id = make_id(sslot, b->orders[sslot].gen);
        int buy_done = fill == b->orders[bslot].qty;
        int sell_done = fill == b->orders[sslot].qty;
        reduce_order(b, bslot, fill);
        reduce_order(b, sslot, fill);
        /* Auction print: aggressor_side 0, taker is the buyer, maker the seller. */
        record_trade(b, ts, best_p, fill, 0, buy_agent, sell_agent, buy_id, sell_id);
        if (buy_done) {
            if (bnext >= 0) {
                bslot = bnext;
            } else {
                b_price = seek_bid(b, b_price - 1, best_p);
                bslot = (b_price >= 0) ? b->bid_head[b_price] : LOB_NIL;
            }
        }
        if (sell_done) {
            if (snext >= 0) {
                sslot = snext;
            } else {
                s_price = seek_ask(b, s_price + 1, best_p);
                sslot = (s_price >= 0) ? b->ask_head[s_price] : LOB_NIL;
            }
        }
    }
    cancel_level(b, 1, LOB_PRICE_MAX, &out->n_moo_cancelled);
    cancel_level(b, -1, 0, &out->n_moo_cancelled);
    b->halted = 0;
    recompute_touch(b);
    out->status = 0;
    out->price = best_p;
    out->n_trades = b->log_n;
    out->truncated = b->truncated;
}

static int walk_ok(const Book *b, const int32_t *head, const int64_t *qty, int32_t *n_out, int64_t *q_out) {
    int32_t n = 0;
    int64_t q = 0;
    uint8_t *seen = (uint8_t *)calloc((size_t)LOB_MAX_ORDERS, 1);
    if (seen == NULL) {
        return 0;
    }
    int ok = 1;
    for (int p = 0; p <= LOB_PRICE_MAX; p++) {
        int32_t slot = head[p];
        int32_t guard = 0;
        int64_t level = 0;
        int32_t prev = LOB_NIL;
        while (slot >= 0) {
            if (slot >= b->n_high || seen[slot] || !b->orders[slot].alive ||
                b->orders[slot].price != p || guard++ > b->n_live + 1) {
                ok = 0;
                break;
            }
            if (b->orders[slot].prev != prev) {
                ok = 0;
                break;
            }
            seen[slot] = 1;
            level += b->orders[slot].qty;
            n++;
            prev = slot;
            slot = b->orders[slot].next;
        }
        if (!ok) {
            break;
        }
        if (level != qty[p]) {
            ok = 0;
            break;
        }
        q += level;
        int32_t tail_expect = prev;
        if (b->bid_head == head || b->ask_head == head) {
            const int32_t *tail = (head == b->bid_head) ? b->bid_tail : b->ask_tail;
            if (tail[p] != tail_expect) {
                ok = 0;
                break;
            }
        }
    }
    free(seen);
    *n_out = n;
    *q_out = q;
    return ok;
}

void lob_audit(const Book *b, Audit *out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (b == NULL || b->magic != LOB_MAGIC) {
        return;
    }
    int32_t n_bid = 0;
    int32_t n_ask = 0;
    int64_t q_bid = 0;
    int64_t q_ask = 0;
    int ok_bid = walk_ok(b, b->bid_head, b->bid_qty, &n_bid, &q_bid);
    int ok_ask = walk_ok(b, b->ask_head, b->ask_qty, &n_ask, &q_ask);
    int32_t expect_bid = LOB_NIL;
    int32_t expect_ask = LOB_NIL;
    for (int p = LOB_PRICE_MAX - 1; p >= 1; p--) {
        if (b->bid_qty[p] > 0 && b->bid_head[p] >= 0) {
            expect_bid = p;
            break;
        }
    }
    for (int p = 1; p <= LOB_PRICE_MAX - 1; p++) {
        if (b->ask_qty[p] > 0 && b->ask_head[p] >= 0) {
            expect_ask = p;
            break;
        }
    }
    int touch_ok = (b->best_bid == expect_bid) && (b->best_ask == expect_ask);
    int count_ok = (n_bid + n_ask) == b->n_live;
    int crossed = (!b->halted && b->best_bid >= 0 && b->best_ask >= 0 && b->best_bid >= b->best_ask);
    out->bid_qty = q_bid;
    out->ask_qty = q_ask;
    out->trade_qty = b->trade_qty;
    out->trade_notional = b->trade_notional;
    out->checksum = (int64_t)b->checksum;
    out->n_bid_orders = n_bid;
    out->n_ask_orders = n_ask;
    out->n_live = b->n_live;
    out->best_bid = b->best_bid;
    out->best_ask = b->best_ask;
    out->crossed = crossed ? 1 : 0;
    out->halted = b->halted;
    out->n_trades = b->n_trades;
    out->list_ok = (ok_bid && ok_ask && touch_ok && count_ok && !crossed) ? 1 : 0;
}

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static void seed_ladder(Book *b, int32_t mid) {
    for (int i = 1; i <= 30; i++) {
        Event ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = EV_LIMIT;
        ev.qty = 40;
        ev.agent = 7;
        ev.side = 1;
        ev.price = mid - i;
        EventResult result;
        lob_submit(b, &ev, &result);
        ev.side = -1;
        ev.price = mid + i;
        lob_submit(b, &ev, &result);
    }
}

static int64_t mono_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return (ticks.QuadPart / frequency.QuadPart) * 1000000000LL
        + (ticks.QuadPart % frequency.QuadPart) * 1000000000LL / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

int lob_bench(int64_t n_events, uint64_t seed, BenchResult *out) {
    if (out == NULL) {
        return 2;
    }
    memset(out, 0, sizeof(*out));
    if (n_events < 1 || n_events > 20000000LL) {
        return 2;
    }
    if (seed == 0) {
        seed = 0x9E3779B97F4A7C15ULL;
    }
    Book *b = lob_new();
    if (b == NULL) {
        return 3;
    }
    b->record = 0;
    const int32_t mid = 50000;
    seed_ladder(b, mid);
    Event *events = (Event *)malloc((size_t)n_events * sizeof(Event));
    if (events == NULL) {
        lob_free(b);
        return 3;
    }
    int64_t t0 = mono_ns();
    uint64_t state = seed;
    int32_t n_rejects = 0;
    for (int64_t i = 0; i < n_events; i++) {
        uint64_t r = xorshift64(&state);
        Event ev;
        memset(&ev, 0, sizeof(ev));
        ev.ts = i + 1;
        ev.agent = 2;
        int kind = (int)(r % 100ULL);
        int side = (r & 1024ULL) ? 1 : -1;
        if (kind < 15) {
            ev.type = EV_MARKET;
            ev.side = side;
            ev.qty = 1 + (int32_t)((r >> 12) % 8ULL);
        } else if (kind < 30) {
            ev.type = EV_LIMIT;
            ev.side = side;
            int32_t extra = (int32_t)((r >> 20) % 3ULL);
            if (side > 0 && b->best_ask > 0) {
                ev.price = b->best_ask + extra;
            } else if (side < 0 && b->best_bid > 0) {
                ev.price = b->best_bid - extra;
            } else {
                ev.price = mid;
            }
            if (ev.price < 1) {
                ev.price = 1;
            }
            if (ev.price > LOB_PRICE_MAX - 1) {
                ev.price = LOB_PRICE_MAX - 1;
            }
            ev.qty = 1 + (int32_t)((r >> 30) % 5ULL);
        } else if (kind < 60) {
            int32_t px = (side > 0) ? b->best_bid : b->best_ask;
            int32_t slot = LOB_NIL;
            if (px > 0) {
                slot = (side > 0) ? b->bid_head[px] : b->ask_head[px];
            }
            if (px > 0 && slot >= 0) {
                ev.type = EV_CANCEL;
                ev.order_id = make_id(slot, b->orders[slot].gen);
            } else {
                ev.type = EV_LIMIT;
                ev.side = side;
                ev.price = (side > 0) ? mid - 2 : mid + 2;
                ev.qty = 1;
            }
        } else {
            ev.type = EV_LIMIT;
            ev.side = side;
            int32_t off = 1 + (int32_t)((r >> 16) % 8ULL);
            if (side > 0) {
                int32_t touch = b->best_bid > 0 ? b->best_bid : mid;
                ev.price = touch - off;
                if (b->best_ask > 0 && ev.price >= b->best_ask) {
                    ev.price = b->best_ask - 1;
                }
            } else {
                int32_t touch = b->best_ask > 0 ? b->best_ask : mid;
                ev.price = touch + off;
                if (b->best_bid > 0 && ev.price <= b->best_bid) {
                    ev.price = b->best_bid + 1;
                }
            }
            if (ev.price < 1) {
                ev.price = 1;
            }
            if (ev.price > LOB_PRICE_MAX - 1) {
                ev.price = LOB_PRICE_MAX - 1;
            }
            ev.qty = 1 + (int32_t)((r >> 28) % 5ULL);
        }
        events[i] = ev;
        EventResult result;
        lob_submit(b, &ev, &result);
        if (result.status == ST_REJECTED) {
            n_rejects++;
        }
    }
    int64_t t1 = mono_ns();
    uint64_t checksum_1 = b->checksum;
    int32_t trades_1 = b->n_trades;
    lob_free(b);

    Book *replay = lob_new();
    if (replay == NULL) {
        free(events);
        return 3;
    }
    replay->record = 0;
    /* calloc is demand-paged. Touch the order arena before the timer so the
     * measurement is matching work, not first-fault cost. Documented in the
     * Python report.
     */
    memset(replay->orders, 0, (size_t)LOB_MAX_ORDERS * sizeof(Order));
    seed_ladder(replay, mid);
    int64_t t2 = mono_ns();
    for (int64_t i = 0; i < n_events; i++) {
        EventResult result;
        lob_submit(replay, &events[i], &result);
    }
    int64_t t3 = mono_ns();
    free(events);
    if (replay->checksum != checksum_1 || replay->n_trades != trades_1) {
        lob_free(replay);
        return 4;
    }
    Audit audit;
    lob_audit(replay, &audit);
    out->n_events = n_events;
    out->n_trades = replay->n_trades;
    out->checksum = (int64_t)replay->checksum;
    out->elapsed_match_ns = t3 - t2;
    out->elapsed_generate_ns = t1 - t0;
    out->list_ok = audit.list_ok;
    out->n_rejects = n_rejects;
    lob_free(replay);
    return 0;
}
