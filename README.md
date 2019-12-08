# automark

AutoMark! The automatic marking feedback tool that you need to write an ACL paper due in four days\*!

\*not yet completed


## Training

`python3.5 -m automark train configs/humanmt.yml`

## Generation

Example:
`python3.5 -m automark generate configs/humanmt.yml data/head.markings.tok.en data/head.markings.tok.hyp my_hyps.txt`