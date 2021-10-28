set.seed(1)
sample(10)

#6.5.1 Best Subset Selection

library(ISLR)
fix(Hitters)
names(Hitters)

dim(Hitters)

sum(is.na(Hitters$Salary))

#The na.omit() function removes all of the rows that have missing values in any variable.
Hitters =na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

#The regsubsets() function (part of the leaps library) performs best sub- regsubsets()
#set selection by identifying the best model that contains a given number
#of predictors, where best is quantified using RSS

library(leaps)
regfit.full=regsubsets(Salary???.,Hitters)
summary(regfit.full)
names(regfit.full)
#[1] "np"        "nrbar"     "d"         "rbar"      "thetab"    "first"     "last"      "vorder"    "tol"      
#[10] "rss"       "bound"     "nvmax"     "ress"      "ir"        "nbest"     "lopt"      "il"        "ier"      
#[19] "xnames"    "method"    "force.in"  "force.out" "sserr"     "intercept" "lindep"    "nullrss"   "nn"       
#[28] "call" 
#For instance, this output indicates that the best two-variable model
#contains only Hits and CRBI. WHY NOT PutOuts VARIABLE?

regfit.full=regsubsets(Salary???.,data=Hitters ,nvmax=19)
reg.summary =summary(regfit.full)
#summary(regfit.full)#1 subsets of each size up to 19
names(reg.summary)#[1] "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj" 

reg.summary$rsq
#For instance, we see that the R2 statistic increases from 32 %, when only
#one variable is included in the model, to almost 55 %, when all variables
#are included. As expected, the R2 statistic increases monotonically as more
#variables are included.
#[1] 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146 0.5141227 0.5285569 0.5346124 0.5404950 0.5426153 0.5436302 0.5444570
#[14] 0.5452164 0.5454692 0.5457656 0.5459518 0.5460945 0.5461159  # 19 VALUES - ALL VAR.

#Plotting RSS, adjusted R2, Cp, and BIC for all of the models

#RSS
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",
       type="l")

#Adjusted R2
plot(reg.summary$adjr2 ,xlab="Number of Variables ",
       ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch =20)

#Cp 

plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp",type="l")
which.min(reg.summary$cp)

points (10,reg.summary$cp [10], col ="red",cex=2,pch =20)

#BIC statistics

which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables ",ylab="BIC", type="l")
points(6,reg.summary$bic [6],col="red",cex=2,pch =20)

#The regsubsets() function has a built-in plot() command which can
#be used to display the selected variables for the best model with a given
#number of predictors,
plot(regfit.full ,scale="r2")
plot(regfit.full ,scale="adjr2")
plot(regfit.full ,scale="Cp")
plot(regfit.full ,scale="bic")
#The top row of each plot contains a black square for each variable selected
#according to the optimal model associated with that statistic.
#For instance,we see that several models share a BIC close to ???150.However, the model
#with the lowest BIC is the six-variable model that contains only AtBat,
#Hits, Walks, CRBI, DivisionW, and PutOuts. We can use the coef() function
#to see the coefficient estimates associated with this model.

coef(regfit.full ,6)

#6.5.2 Forward and Backward Stepwise Selection

regfit.fwd=regsubsets(Salary???.,data=Hitters, nvmax=19,method ="forward")
summary(regfit.fwd)

regfit.bwd=regsubsets(Salary???.,data=Hitters, nvmax=19,method ="backward")
summary(regfit.bwd)

#However, the best seven-variable models identified by forward stepwise selection, 
#backward stepwise selection, and best subset selection are different.

coef(regfit.full ,7)

coef(regfit.fwd ,7)

coef(regfit.bwd ,7)

#6.5.3 Choosing Among Models Using the Validation Set Approach and Cross-Validation

#see pg 248

set.seed(1)
train=sample(c(TRUE ,FALSE), nrow(Hitters),rep=TRUE)
test=(!train)

#Now, we apply regsubsets() to the training set in order to perform best subset selection.
regfit.best=regsubsets(Salary???.,data=Hitters[train ,], nvmax=19)

test.mat=model.matrix(Salary???.,data=Hitters [test ,])

val.errors =rep(NA ,19)
for(i in 1:19){
coefi=coef(regfit.best ,id=i)
pred=test.mat[,names(coefi)]%*%coefi
val.errors[i]=mean(( Hitters$Salary[test]-pred)^2)}

val.errors

which.min(val.errors)

coef(regfit.best ,10)

predict.regsubsets =function (object , newdata ,id ,...){
form=as.formula (object$call [[2]])
mat=model.matrix(form ,newdata )
coefi=coef(object ,id=id)
xvars=names(coefi)
mat[,xvars]%*%coefi}

regfit.best=regsubsets(Salary???.,data=Hitters ,nvmax=19)
coef(regfit.best ,10)

k=10
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors =matrix(NA,k,19, dimnames =list(NULL, paste(1:19)))

for(j in 1:k){best.fit=regsubsets (Salary???.,data=Hitters [folds!=j,],nvmax=19)
for(i in 1:19){pred=predict (best.fit ,Hitters [folds ==j,],id=i)
cv.errors[j,i]= mean( ( Hitters$Salary[ folds==j]-pred)^2)}}

mean.cv.errors=apply(cv.errors ,2, mean)
mean.cv.errors

par(mfrow=c(1,1))
plot(mean.cv.errors ,type="b")

reg.best=regsubsets(Salary???.,data=Hitters , nvmax=19)
coef(reg.best ,11)

#6.6 Lab 2: Ridge Regression and the Lasso

x=model.matrix(Salary???.,Hitters )[,-1]
y=Hitters$Salary

#6.6.1 Ridge Regression
install.packages("glmnet", dependencies=TRUE)
installed.packages("glmnet")
library(glmnet)
grid=10^seq(10,-2, length =100)
ridge.mod=glmnet(x,y,alpha=0, lambda=grid)

dim(coef(ridge.mod))

ridge.mod$lambda [50]
coef(ridge.mod)[ ,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda [60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

predict(ridge.mod,s=50,type="coefficients")[1:20,]

set.seed(1)
train=sample (1: nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

ridge.mod=glmnet(x[train ,],y[ train],alpha=0, lambda =grid,thresh =1e-12)
ridge.pred=predict(ridge.mod ,s=4, newx=x[test ,])
mean((ridge.pred -y.test)^2)

mean((mean(y[train])-y.test)^2)

ridge.pred=predict(ridge.mod ,s=1e10 ,newx=x[test ,])
mean((ridge.pred -y.test)^2)

ridge.pred=predict(ridge.mod,s=0.01, newx=x[test,], exact=T)
mean((ridge.pred -y.test)^2)
#When s < 0.01, the following error arises:
#Error: used coef.glmnet() or predict.glmnet() with exact=TRUE so must in 
#addition supply original argument(s) x and y in order to safely rerun glmnet
#So here I use s = 0.01 instead of s = 0 as a workaround.
#https://rpubs.com/leechau/isl-ch6-lab

lm(y???x, subset=train)
predict (ridge.mod ,s=0.01,exact=T,type="coefficients")[1:20,]

set.seed(1)
cv.out=cv.glmnet(x[train ,],y[ train],alpha=1)#alpha it was 0
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
format(round(bestlam,0)) # to keep out decimals

ridge.pred=predict(ridge.mod ,s=bestlam ,newx=x[test ,])
mean((ridge.pred -y.test)^2)

out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s= bestlam)[1:20,]

#6.6.2 The Lasso
lasso.mod=glmnet(x[train ,],y[ train],alpha=1, lambda =grid)
plot(lasso.mod)

set.seed(1)
cv.out=cv.glmnet(x[train ,],y[ train],alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
lasso.pred=predict(lasso.mod ,s=bestlam ,newx=x[test ,])
mean((lasso.pred -y.test)^2)

out=glmnet(x,y,alpha=1, lambda=grid)
lasso.coef=predict(out ,type="coefficients",s= bestlam)[1:20,]
lasso.coef

#6.7 Lab 3: PCR and PLS Regression
#6.7.1 Principal Components Regression

library(pls)
Hitters <- na.omit(Hitters)
set.seed(2)
pcr.fit=pcr(Salary~., data=Hitters, scale= TRUE, validation = "CV")
summary(pcr.fit)

validationplot(pcr.fit ,val.type="MSEP")

set.seed(1)
pcr.fit=pcr(Salary???., data=Hitters , subset=train ,scale=TRUE,validation ="CV")
validationplot(pcr.fit ,val.type="MSEP") #see the plot

pcr.pred=predict(pcr.fit ,x[test ,],ncomp =7)
mean((pcr.pred-y.test)^2)

pcr.fit=pcr(y???x,scale=TRUE ,ncomp=7)
summary (pcr.fit)

#6.7.2 Partial Least Squares
set.seed(1)
pls.fit=plsr(Salary???., data=Hitters , subset=train , scale=TRUE ,
               validation ="CV")
summary (pls.fit)

validationplot(pls.fit ,val.type="MSEP")

pls.pred=predict(pls.fit ,x[test ,],ncomp =2)
mean((pls.pred -y.test)^2)

pls.fit=plsr(Salary???., data=Hitters , scale=TRUE , ncomp=2)
summary (pls.fit)

