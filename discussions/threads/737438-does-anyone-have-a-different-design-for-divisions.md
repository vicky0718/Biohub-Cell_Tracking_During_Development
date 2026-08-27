# does anyone have a different design for divisions

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438
- **Topic id**: 737438
- **Author**: kevin park (CONTRIBUTOR)
- **Posted**: 2026-08-25T14:20:22.701260400Z
- **Votes**: 1
- **Comments**: 0

---

## Opening post

hey im kinda stuck and wanna know if ppl have a different design to me especially in a few areas.

my linker is basically 1:1 right now. hungarian assignment frame to frame. then i add divisions in a totally separate step after on top of the result. so the division decision never actually sees the model. its js a hand written rule with distance and image checks.

my divJ is around 0.12 which is way below what some ppl report. edge side is ok. its the divisions.

the thing i measured that surprised me. the ones im missing arent random at all. theyre almost all in crowded frames. like 3x the node count of the ones i get right and closer neighbours. and when i checked why each one failed basically none of them fail for js one reason. they fail 2 or 3 checks at once. so loosening any single threshold recovers nothing. it js adds false ones.

so the areas where i wanna know if ur design is different.

where the division decision lives. do u let a cell have 2 children inside the assignment itself w some learned prob. or do u bolt it on after like me. if its inside was it worth the rewrite.

dense regions. is there a design that actually handles crowded frames. or does everyone js lose those and accept it.

high divJ. for ppl getting way more than me is it a better model on the same structure. or is the structure itself different.

not asking for anyones solution. js wanna know if im stuck on a design everyone else already moved past. thanks

---

## Comments (0)

*(none)*
