library(randomForest)
library(e1071)
library(glmnet)
library(plotly)

###########################
####Deep's LP Functions####
###########################

#This is an edited version of the LP.Score.fun() that fills in NAs for 
#discrete data when m is larger than the number of unique data points
LP.Score.fun2 <- function(x,m){ 
  u <- (rank(x,ties.method = c("average")) - .5)/length(x) 
  n <- min(length(unique(u ))-1, m )
  S.mat <- as.matrix(poly(u ,df=n)) 
  if (ncol(S.mat) < m) {
    cols <- matrix(rep(0, (length(x)*(m-n))), nrow= length(x))
    S.mat <- cbind(S.mat, cols)
  }
  return(as.matrix(scale(S.mat)))
}


LP.smooth <- function(CR,n,method){ ###--"AIC" or "BIC"
  CR.s <- sort(CR^2,decreasing=TRUE,index=TRUE)$x
  aa <- rep(0,length(CR.s))
  if(method=="AIC"){ penalty <- 2}
  if(method=="BIC"){ penalty <- log(n)}
  aa[1] <- CR.s[1] - penalty/n
  if(aa[1]< 0){ return(rep(0,length(CR))) }
  for(i in 2: length(CR.s)){
    aa[i] <- aa[(i-1)] + (CR.s[i] - penalty/n)
  }
  #plot(aa,type="b",ylab=method,cex.axis=1.2,cex.lab=1.2)
  CR[CR^2<CR.s[which(aa==max(aa))]] <- 0
  return(CR)
}

#---------------------------------------------------------------------------#

#Transforms Matrix
LPT <- function(x, m) {
  xMat <- as.data.frame(x)
  colnames(xMat) <- seq(1:ncol(xMat))
  tvals <- lapply(xMat, LP.Score.fun2, m=m)
  for (i in 1:ncol(xMat)) {
    colnames(tvals[[i]])<- paste0(colnames(xMat)[i], "_", 1:m)
  }
  tmat <- do.call(cbind, tvals)
  return(tmat)
}



#LP Logistic
#Params can be other parameters in the cv.glmnet model
LP.logistic <- function(x, y, m, smooth=NULL, params) {
  t_train <- LPT(x, m)
  if (is.factor(y)) {
    LPcorr <- cor(as.numeric(levels(y))[y], t_train)
  } else {
    LPcorr <- cor(y, t_train)
  }
  if (is.null(smooth)) {
    reducedT <- t_train
  } else if (smooth=="AIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "AIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  } else if (smooth=="BIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "BIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  }
  
  if (missing(params)) {
    logistic <- cv.glmnet(reducedT, y, family="multinomial")
  } else {
    logistic <- do.call(cv.glmnet, c(list(x=reducedT, y=y, family="multinomial"), params))
  }
  output <- list("model" = logistic, "Tmatrix" = t_train,  
                 "SmoothedT" = reducedT)
  return(output)
}

#LP SVM 
#Can set tuning parameters in tune.svm() function
#Can set parameters in svm() function
LP.svm <- function(x, y, m, smooth=NULL, tune=FALSE, tuneparams, params) {
  t_train <- LPT(x, m)
  if (is.factor(y)) {
    LPcorr <- cor(as.numeric(levels(y))[y], t_train)
  } else {
    LPcorr <- cor(y, t_train)
  }
  if (is.null(smooth)) {
    reducedT <- t_train
  } else if (smooth=="AIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "AIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  } else if (smooth=="BIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "BIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  }
  
  if (missing(params)) {
    if (tune==TRUE) {
      tunes <- do.call(tune.svm, c(list(x=reducedT, y=y), tuneparams))
      svm <- svm(x=reducedT, y=y, gamma=tunes$best.parameters[[1]],
                 cost=tunes$best.parameters[[2]])
    } else{
      svm <- svm(x=reducedT, y=y)
    }
  } else {
    if (tune==TRUE) {
      tunes <- do.call(tune.svm, c(list(x=reducedT, y=y), tuneparams))
      svm <- do.call(svm, c(list(x=reducedT, y=y, gamma=tunes$best.parameters[[1]],
                                 cost=tunes$best.parameters[[2]]), params))
    } else {
      svm <- do.call(svm, c(list(x=reducedT, y=y), params))
    }
    
  }
  output <- list("model" = svm, "Tmatrix" = t_train,  
                 "SmoothedT" = reducedT)
  return(output)
}



#LP Random Forest
LP.rf <- function(x, y, m, smooth=NULL, params) {
  t_train <- LPT(x, m)
  if (is.factor(y)) {
    LPcorr <- cor(as.numeric(levels(y))[y], t_train)
  } else {
    LPcorr <- cor(y, t_train)
  }
  if (is.null(smooth)) {
    reducedT <- t_train
  } else if (smooth=="AIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "AIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  } else if (smooth=="BIC") {
    Tsmooth <- LP.smooth(LPcorr, length(y), "BIC")
    reducedT <- t_train[, which(Tsmooth != 0)]
  }
  
  if (missing(params)) {
    forest <- randomForest(reducedT, y)
  } else {
    forest <- do.call(randomForest, c(list(x=reducedT, y=y), params))
  }
  output <- list("model" = forest, "Tmatrix" = t_train,  
                 "SmoothedT" = reducedT)
  return(output)
}


#LP prediction function to work with LP.logistic, LP.svm, and LP.rf
#Be careful about setting the type for different models
LP.predict <- function(model, newdata, m, type="class", params) {
  t_test <- LPT(newdata, m)
  testsmooth <- t_test[, colnames(model$SmoothedT)]
  if (missing(params)) {
    if (class(model$model) == "svm") {
      prediction <- predict(model$model, testsmooth)
    } else {
      prediction <- predict(model$model, testsmooth,
                            type)
    }
  } else {
    if (class(model$model) == "svm") {
      prediction <- do.call(predict, c(list(object=model$model, as.matrix(testsmooth)),
                                       params))
    } else {
      prediction <- do.call(predict, c(list(object=model$model, as.matrix(testsmooth),
                                            type=type),params))
    }
  }
  output <- list("Tmatrix" = t_test, "Prediction" = prediction, 
                 "SmoothedT" = testsmooth)
  return(output)
}

#requires plotly
#Will be updated in the future so that one could choose "AIC", "BIC"
#Also other sorting options, and other graphs
LP.plot <- function(x, y, m, smooth=TRUE) {
  t <- LPT(x, m)
  LPcorr <- cor(y, t)
  if (smooth==TRUE) {
  Tsmooth <- LP.smooth(LPcorr, length(y), "BIC")
  reducedT <- t[, which(Tsmooth != 0)]
  reducedsplit <- strsplit(colnames(reducedT), "_")
  tmatsplit <- strsplit(colnames(t), "_")
  tmatnames <- sapply(tmatsplit, "[", 1)
  reducednames <- sapply(reducedsplit, "[", 1)
  sigfeat <- t[,tmatnames %in% reducednames]
  } else {
    sigfeat <- t
    tmatsplit <- strsplit(colnames(t), "_")
    tmatnames <- sapply(tmatsplit, "[", 1)
    reducednames <- tmatnames
  }
  sigcor <- cor(y, sigfeat)
  matcor <- matrix(sigcor^2, nrow=m, ncol=length(unique(reducednames)))
  matcor[is.na(matcor)] <- 0
  colnames(matcor) <- as.numeric(unique(reducednames))
  sorted <- matcor[,order(colSums(matcor), decreasing=TRUE)]
  xa <- list(
    
    title = "LP Moments"
  )
  ya <- list(
    showticklabels = FALSE,
    title = "Features"
  )
  
  c <- list(c(0, 'azure'),c(.5, '#9ecae1'),c(1,'#3182bd'))
  bar <- list(
    title = "Correlation"
  )

  plot_ly(z=t(sorted), x=1:m, y=colnames(x[,as.numeric(colnames(sorted))]),
          type="heatmap", hoverinfo = "x+y+z", colorscale=c, colorbar = bar) %>%
    layout(xaxis = xa, yaxis = ya)
  
  
}




