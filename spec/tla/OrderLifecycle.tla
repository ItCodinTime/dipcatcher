------------------------------ MODULE OrderLifecycle ------------------------------
\* Order lifecycle of one simulated order.
\*
\* Statuses: absent, new (working, unfilled; the broker's ACKED), partial
\* (working residual), filled, canceled, rejected, expired.
\*
\* Safety (checked with crashes, amends, duplicate fills, late fills):
\*   never overfilled, terminal shape, fill ids apply at most once,
\*   a durable terminal snapshot stays terminal, cancel/expire/reject
\*   does not fill in the same step, a committed terminal order stays put.
\*
\* Liveness (LivenessSpec): no crashes and no amends, weak fairness of
\* Resolve. A working order eventually becomes terminal. Crashes or
\* repeated upward amends can postpone that; those behaviors are outside
\* LivenessSpec on purpose.
\*
\* Quantities and fill ids are bounded integers so TLC can exhaust the
\* state space. The Python twin in quant_fund.formal.order_lifecycle is
\* the same relation over positive reals and arbitrary ids.

EXTENDS Integers

CONSTANTS MaxQty, MaxFillId

ASSUME MaxQty \in Nat \ {0} /\ MaxFillId \in Nat \ {0}

Qty == 1..MaxQty
FillIds == 1..MaxFillId
Status == {"absent", "new", "partial", "filled", "canceled", "rejected", "expired"}
Terminal == {"filled", "canceled", "rejected", "expired"}
Working == {"new", "partial"}

VARIABLES status, qty, filled, applied, fqty,
          s_status, s_qty, s_filled, s_applied, s_fqty

vars == <<status, qty, filled, applied, fqty,
          s_status, s_qty, s_filled, s_applied, s_fqty>>

\* Sum of fq[id] over ids. Minimum-element recursion keeps the sum a function.
RECURSIVE SumOf(_, _)
SumOf(ids, fq) ==
  IF ids = {} THEN 0
  ELSE LET i == CHOOSE x \in ids : \A y \in ids : x <= y
       IN fq[i] + SumOf(ids \ {i}, fq)

TypeOK ==
  /\ status \in Status
  /\ s_status \in Status
  /\ qty \in 0..MaxQty
  /\ s_qty \in 0..MaxQty
  /\ filled \in 0..MaxQty
  /\ s_filled \in 0..MaxQty
  /\ applied \subseteq FillIds
  /\ s_applied \subseteq FillIds
  /\ fqty \in [FillIds -> 0..MaxQty]
  /\ s_fqty \in [FillIds -> 0..MaxQty]

Init ==
  /\ status = "absent"
  /\ qty = 0
  /\ filled = 0
  /\ applied = {}
  /\ fqty = [i \in FillIds |-> 0]
  /\ s_status = "absent"
  /\ s_qty = 0
  /\ s_filled = 0
  /\ s_applied = {}
  /\ s_fqty = [i \in FillIds |-> 0]

Submit(q) ==
  /\ status = "absent"
  /\ q \in Qty
  /\ status' = "new"
  /\ qty' = q
  /\ UNCHANGED <<filled, applied, fqty, s_status, s_qty, s_filled, s_applied, s_fqty>>

Reject ==
  /\ status \in Working
  /\ status' = "rejected"
  /\ UNCHANGED <<qty, filled, applied, fqty, s_status, s_qty, s_filled, s_applied, s_fqty>>

Cancel ==
  /\ status \in Working
  /\ status' = "canceled"
  /\ UNCHANGED <<qty, filled, applied, fqty, s_status, s_qty, s_filled, s_applied, s_fqty>>

Expire ==
  /\ status \in Working
  /\ status' = "expired"
  /\ UNCHANGED <<qty, filled, applied, fqty, s_status, s_qty, s_filled, s_applied, s_fqty>>

\* rest = TRUE keeps a shortfall working (limit residual).
\* rest = FALSE closes the order (IOC), even when filled' < qty.
Fill(id, q, rest) ==
  /\ status \in Working
  /\ id \in FillIds
  /\ id \notin applied
  /\ q \in Qty
  /\ filled + q <= qty
  /\ applied' = applied \cup {id}
  /\ fqty' = [fqty EXCEPT ![id] = q]
  /\ filled' = filled + q
  /\ status' = IF rest /\ filled + q < qty THEN "partial" ELSE "filled"
  /\ UNCHANGED <<qty, s_status, s_qty, s_filled, s_applied, s_fqty>>

\* Second application of a fill id is a stutter.
DuplicateFill(id) ==
  /\ id \in applied
  /\ UNCHANGED vars

\* A late fill against a terminal order is ignored.
LateFill(id) ==
  /\ status \in Terminal
  /\ id \in FillIds
  /\ UNCHANGED vars

\* q is the new total authorization and must stay strictly above filled.
Amend(q) ==
  /\ status \in Working
  /\ q \in Qty
  /\ q > filled
  /\ q /= qty
  /\ qty' = q
  /\ status' = IF filled = 0 THEN "new" ELSE "partial"
  /\ UNCHANGED <<filled, applied, fqty, s_status, s_qty, s_filled, s_applied, s_fqty>>

Checkpoint ==
  /\ \/ s_status /= status
     \/ s_qty /= qty
     \/ s_filled /= filled
     \/ s_applied /= applied
     \/ s_fqty /= fqty
  /\ s_status' = status
  /\ s_qty' = qty
  /\ s_filled' = filled
  /\ s_applied' = applied
  /\ s_fqty' = fqty
  /\ UNCHANGED <<status, qty, filled, applied, fqty>>

\* Drop uncommitted volatile state. Durable fills stay.
Crash ==
  /\ \/ status /= s_status
     \/ qty /= s_qty
     \/ filled /= s_filled
     \/ applied /= s_applied
     \/ fqty /= s_fqty
  /\ status' = s_status
  /\ qty' = s_qty
  /\ filled' = s_filled
  /\ applied' = s_applied
  /\ fqty' = s_fqty
  /\ UNCHANGED <<s_status, s_qty, s_filled, s_applied, s_fqty>>

Next ==
  \/ \E q \in Qty : Submit(q)
  \/ Reject
  \/ Cancel
  \/ Expire
  \/ \E id \in FillIds, q \in Qty, rest \in BOOLEAN : Fill(id, q, rest)
  \/ \E id \in FillIds : DuplicateFill(id)
  \/ \E id \in FillIds : LateFill(id)
  \/ \E q \in Qty : Amend(q)
  \/ Checkpoint
  \/ Crash

\* Liveness environment: crashes stop, and authorization is not amended upward.
NextLive ==
  \/ \E q \in Qty : Submit(q)
  \/ Reject
  \/ Cancel
  \/ Expire
  \/ \E id \in FillIds, q \in Qty, rest \in BOOLEAN : Fill(id, q, rest)
  \/ \E id \in FillIds : DuplicateFill(id)
  \/ \E id \in FillIds : LateFill(id)
  \/ Checkpoint

Resolve ==
  \/ Reject
  \/ Cancel
  \/ Expire
  \/ \E id \in FillIds, q \in Qty, rest \in BOOLEAN : Fill(id, q, rest)

SafetySpec == Init /\ [][Next]_vars
LivenessSpec == Init /\ [][NextLive]_vars /\ WF_vars(Resolve)

NeverOverfilled ==
  /\ filled <= qty
  /\ s_filled <= s_qty

TerminalShape ==
  /\ status = "absent" => filled = 0 /\ qty = 0 /\ applied = {}
  /\ status = "new" => filled = 0 /\ qty \in Qty
  /\ status = "partial" => filled > 0 /\ filled < qty
  /\ status = "filled" => filled > 0 /\ filled <= qty
  /\ status \in {"canceled", "rejected", "expired"} => filled <= qty
  /\ s_status = "absent" => s_filled = 0 /\ s_qty = 0 /\ s_applied = {}
  /\ s_status = "new" => s_filled = 0 /\ s_qty \in Qty
  /\ s_status = "partial" => s_filled > 0 /\ s_filled < s_qty
  /\ s_status = "filled" => s_filled > 0 /\ s_filled <= s_qty
  /\ s_status \in {"canceled", "rejected", "expired"} => s_filled <= s_qty

FillAccounting ==
  /\ filled = SumOf(applied, fqty)
  /\ s_filled = SumOf(s_applied, s_fqty)
  /\ \A id \in applied : fqty[id] > 0
  /\ \A id \in FillIds \ applied : fqty[id] = 0
  /\ \A id \in s_applied : s_fqty[id] > 0
  /\ \A id \in FillIds \ s_applied : s_fqty[id] = 0

CheckpointPrefix ==
  /\ s_applied \subseteq applied
  /\ s_filled <= filled
  /\ \A id \in s_applied : s_fqty[id] = fqty[id]

\* One step does not both fill and cancel/expire/reject.
\* A durable terminal snapshot never changes.
\* A terminal order that has been checkpointed stays terminal.
\* The durable fill set only grows, and durable filled only grows.
FillDoesNotCancel ==
  filled' > filled => status' \in {"partial", "filled"}
StableTerminalSticky ==
  s_status \in Terminal =>
    (s_status' = s_status /\ s_filled' = s_filled /\ s_applied' = s_applied)
CommittedTerminalSticky ==
  (status \in Terminal /\ status = s_status) =>
    (status' = status /\ filled' = filled)
DurableMonotone ==
  s_applied \subseteq s_applied' /\ s_filled' >= s_filled
ActionSafety ==
  /\ [][FillDoesNotCancel]_vars
  /\ [][StableTerminalSticky]_vars
  /\ [][CommittedTerminalSticky]_vars
  /\ [][DurableMonotone]_vars

EventuallyTerminal ==
  (status \in Working) ~> (status \in Terminal)

=============================================================================
