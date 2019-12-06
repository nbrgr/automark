# automark

AutoMark! The automatic marking feedback tool that you need to write an ACL paper due in four days\*!

\*not yet completed

Notes to Julia:

In my terminal it's printing all of the token ids. I can't find where it's printing that so sorry if your terminal is spammed with a bunch of tensors.

Padding doesn't seem to be working on RawFields, but maybe bucketiterator is doing really good at making sure that examples are the same size.

Loss is going down really quick using Adam but not SGD. Logging frequency was 10 but it still seems odd that Adam was that quick to lower loss within 10 batches.
