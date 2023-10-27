library(exact2x2)
library(effsize)
library(xtable)


res=list(Dataset=c(),McNemar.p=c(),McNemar.OR=c())
d<-"../data/correct_predictions.csv"
t<-read.csv(d)

m=mcnemar.exact(t$isPerfectBlank,t$isPerfectT5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)


m=mcnemar.exact(t$isPerfectTS1,t$isPerfectT5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$isPerfectTS2,t$isPerfectT5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$isPerfectTS3,t$isPerfectT5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)

m=mcnemar.exact(t$isPerfectRF,t$isPerfectT5)
res$Dataset=c(res$Dataset,as.character(d))
res$McNemar.p=c(res$McNemar.p, m$p.value)
res$McNemar.OR=c(res$McNemar.OR,m$estimate)



res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="holm")
print(res)



# print(xtable(res),include.rownames=FALSE)
