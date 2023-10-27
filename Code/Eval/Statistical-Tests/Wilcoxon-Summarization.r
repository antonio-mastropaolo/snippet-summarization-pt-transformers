library(effsize)


res=list(Dataset=c(),Wilcoxon.p=c())


T5Jaccard<-read.csv("../data/METEOR-T5-Jaccard.csv",header=TRUE)
T5TF<-read.csv("../data/METEOR-T5-TF-IDF.csv",header=TRUE)

#p-value < 0.05 to be significant

res$Dataset=c(res$Dataset,as.character("../data/METEOR-T5-Jaccard.csv"))
res$Wilcoxon.p=c(res$Wilcoxon.p,wilcox.test(T5Jaccard$scoreJaccard,T5Jaccard$scoreT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(T5Jaccard$scoreJaccard,T5Jaccard$scoreT5)

res$Dataset=c(res$Dataset,as.character("../data/METEOR-T5-TF-IDF.csv"))
res$Wilcoxon.p=c(res$Wilcoxon.p,wilcox.test(T5TF$scoreTF,T5TF$scoreT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(T5TF$scoreTF,T5TF$scoreT5)


res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p,method="holm")
print(res)





