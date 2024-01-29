*Motivation:* metadata.yaml will not be complete until you add the corresponding FUEL ICs (initial conditions) at the end!\n(found in data\'s ablate config file, e.g. sampleDiffusionFlame.yaml)

**tl;dr for new ablate datasets make new files IC-cfg like sampleDiffusionFlame.yaml**

For this reason you must create an Initial\_Condition\_Config file for a given ablate input config file (by convention it should have the same name, e.g. input\_config: sampleDiffusionFlame.yaml --> IC\_cfg: sampleDiffusionFlame.yaml). The **IC-cfg file should contain 2 fields: 'mechanism'** (relative path to the corresponding mechanism file), **and 'initializers'** which are the Initial Conditions extracted from the original input\_config (extraction is non-trivial & you should check with Matt that you've done it right).

The idea here is even if we cannot automate extraction of the ICs, we can still make a local database of sorts of ICs which have already been extracted so it only needs to be done manually once!
