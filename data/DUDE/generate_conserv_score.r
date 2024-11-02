Args <- commandArgs(TRUE)
path = Args[1]
#print(path)
library(bio3d)
aln <- read.fasta(path)
#print(aln)
score = conserv(x=aln$ali, method="identity", sub.matrix="bio3d",normalize.matrix = FALSE)
write.table(score, file = "../drugVQA-master/data/DUDE/tmpt_identity.txt", row.names = FALSE, col.names = FALSE)
score

