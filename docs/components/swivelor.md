# Swivelor

**Engine id:** `lithokraft:swivelor`  
**Spelling:** Swivelor (never Swiveler).  
**Kind:** Full block, rotary base with a resin-mount top.

![Swivelor](images/swivelor.png)

*CloudAtlas. State shown: Swivelor B, detent 0 (home). Turning, carrying an assembly, the clearance check, and collision are not implemented.*

## What is stated

The Swivelor is a rotary base. A resin-bonded body on top is meant to ride the turn. Rotation is always clockwise from above: 64 then 0 is a full clockwise turn, never a rewind. Detents are 0, 16, 32, and 64, one quarter turn each, with 0 as home. It is stated to check its swept area before turning and to have collision.

The block holds its detent in the state value. Home is 0 on placement.

## Open and unimplemented

- **Which face takes the detent strength is OPEN.** There is no control port. The four side wells are reserved and unassigned: this block reads no input on any face and emits nothing.
- **Connection cap**, and whether this revises an earlier fine-aim spec or is a second mode, is OPEN.
- A read-back that lower targets are reached by wrapping clockwise (16 to 32 to 64 to 0) is unconfirmed.
- **Turning is not implemented**, and neither is carrying an assembly, the clearance check, or collision. Those need the open control face and Resin bonding, neither of which is in. The still on this sheet is the home detent, not a mid-turn pose.
