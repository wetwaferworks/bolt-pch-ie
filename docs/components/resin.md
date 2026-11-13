# Resin

**Engine / item id:** `lithokraft:resin` is the pack's proposed string id (not confirmed). Numeric id `kItemResin` is appended in the item registry.  
**Kind:** Item only. Never a block. The Component Test world gives it a name-only pad; the centre stays empty.

![Resin](images/resin.png)

*CloudAtlas. State shown: Resin B pot item. This is not a resin block. No resin source block is in scope here, and there is no item-model renderer reading the on-disk model yet; the HUD uses the pot icon.*

## What it does in this build

Resin exists so the item can be held and tested. A player inventory holds and stacks it. The HUD has a pot icon. `maxStack` is 512, following the plain material rows (that cap is not a separate ruling). The item was added to the 32-entry item registry (append-only; Resin is offset +30).

Resin places nothing, is not a tool, and nothing in the sim reads it. Bonding, application, removal, and assembly counts against [Pushor](pushor.md) and [Swivelor](swivelor.md) limits do not exist yet.

## Open and unimplemented

- The string id `lithokraft:resin` is a proposal, not confirmed.
- A resin source block is described as existing and out of scope for this item.
- How resin is applied or removed, how it bonds into assemblies, and how those assemblies count against the Pushor limit of 32 and any Swivelor connection cap, are OPEN.
- The on-disk item model (`resin_item`) is present in the art pack; no item renderer reads models yet. The still on this sheet is that pot hero, framed at native resolution, not a placed block.
