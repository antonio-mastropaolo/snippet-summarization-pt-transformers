library(effsize)

################## PRECISION ##################

blankT5<-read.csv("../data/precision-T5-Blank.csv",header=TRUE)
rfT5<-read.csv("../data/precision-T5-RF.csv",header=TRUE)
ts1T5<-read.csv("../data/precision-T5-TS1.csv",header=TRUE)
ts2T5<-read.csv("../data/precision-T5-TS2.csv",header=TRUE)
ts3T5<-read.csv("../data/precision-T5-TS3.csv",header=TRUE)

resPrecision=list(Dataset=c(),Wilcoxon.p=c())

resPrecision$Dataset=c(resPrecision$Dataset,as.character("../data/precision-T5-Blank.csv"))
resPrecision$Wilcoxon.p=c(wilcox.test(blankT5$precisionBlank,blankT5$precisionT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(blankT5$precisionBlank,blankT5$precisionT5)

resPrecision$Dataset=c(resPrecision$Dataset,as.character("../data/precision-T5-RF.csv"))
resPrecision$Wilcoxon.p=c(wilcox.test(rfT5$precisionRF,rfT5$precisionT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(rfT5$precisionRF,rfT5$precisionT5)

resPrecision$Dataset=c(resPrecision$Dataset,as.character("../data/precision-T5-TS1.csv"))
resPrecision$Wilcoxon.p=c(wilcox.test(ts1T5$precisionTS1,ts1T5$precisionT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(ts1T5$precisionTS1,ts1T5$precisionT5)

resPrecision$Dataset=c(resPrecision$Dataset,as.character("../data/precision-T5-TS2.csv"))
resPrecision$Wilcoxon.p=c(wilcox.test(ts2T5$precisionTS2,ts2T5$precisionT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(ts2T5$precisionTS2,ts2T5$precisionT5)

resPrecision$Dataset=c(resPrecision$Dataset,as.character("../data/precision-T5-TS3.csv"))
resPrecision$Wilcoxon.p=c(wilcox.test(ts3T5$precisionTS3,ts3T5$precisionT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(ts3T5$precisionTS3,ts3T5$precisionT5)

resPrecision=data.frame(resPrecision)
resPrecision$Wilcoxon.p=p.adjust(resPrecision$Wilcoxon.p,method="holm")
print(resPrecision)

################## RECALL ##################



recall<-read.csv("../data/recall.csv",header=TRUE)

resRecall=list(Dataset=c(),Wilcoxon.p=c())

resRecall$Dataset=c(resRecall$Dataset,as.character("../data/recall-T5-Blank.csv"))
resRecall$Wilcoxon.p=c(wilcox.test(recall$recallBlank,recall$recallT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(recall$recallBlank,recall$recallT5)

resRecall$Dataset=c(resRecall$Dataset,as.character("../data/recall-T5-RF.csv"))
resRecall$Wilcoxon.p=c(wilcox.test(recall$recallRF,recall$recallT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(recall$recallRF,recall$recallT5)

resRecall$Dataset=c(resRecall$Dataset,as.character("../data/recall-T5-TS1.csv"))
resRecall$Wilcoxon.p=c(wilcox.test(recall$recallTS1,recall$recallT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(recall$recallTS1,recall$recallT5)

resRecall$Dataset=c(resRecall$Dataset,as.character("../data/recall-T5-TS2.csv"))
resRecall$Wilcoxon.p=c(wilcox.test(recall$recallTS2,recall$recallT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(recall$recallTS2,recall$recallT5)

resRecall$Dataset=c(resRecall$Dataset,as.character("../data/recall-T5-TS3.csv"))
resRecall$Wilcoxon.p=c(wilcox.test(recall$recallTS3,recall$recallT5,alternative="two.side",paired=TRUE)$p.value)
cliff.delta(recall$recallTS3,recall$recallT5)

resRecall=data.frame(resRecall)
resRecall$Wilcoxon.p=p.adjust(resRecall$Wilcoxon.p,method="holm")
print(resRecall)




